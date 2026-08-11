import logging
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List

import torch
import torch.nn as nn
import cv2
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import trange

# Must run before mmdet/mmpose, which import mmcv.ops (and thus mmcv._ext) eagerly.
# No-op when a fully compiled mmcv is installed. See the module docstring for why
# an RTX 5090 needs this.
from marmopose.utils.mmcv_ext_shim import install as _install_mmcv_ext_shim
_USING_MMCV_SHIM = _install_mmcv_ext_shim()

from mmdet.apis import inference_detector, init_detector
from mmdet.structures import DetDataSample
from mmdet.utils.misc import get_test_pipeline_cfg
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.structures import PoseDataSample
try:
    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import load_config, get_input_shape
    HAS_MMDEPLOY = True
except ImportError:
    # mmdeploy is only needed for the TensorRT-deployed (*.engine) models.
    # Without it the *.pth models still work, so don't fail the whole import.
    build_task_processor = load_config = get_input_shape = None
    HAS_MMDEPLOY = False
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.logging import MMLogger
MMLogger.get_current_instance().setLevel(logging.ERROR)

from marmopose.config import Config
from marmopose.utils.data_io import save_points_bboxes_2d_h5
from marmopose.utils.torch_compat import legacy_torch_load

logger = logging.getLogger(__name__)

if _USING_MMCV_SHIM:
    logger.info('mmcv._ext not found; using the torchvision-backed NMS shim.')


class Predictor:
    def __init__(self, config: Config, batch_size: int = 4, device: str = None):
        self.init_dir(config)
        self.init_data_cfg(config)

        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu") # TODO: Multiple devices support
        logger.info(f'Predictor device: {self.device} | batch size: {self.batch_size}')

        # mmengine's checkpoint loader predates torch 2.6 and cannot ask for
        # weights_only=False itself.
        with legacy_torch_load():
            self.use_deployed_det = False
            self.build_det_model(config.directory['det_model'])

            self.use_deployed_pose = False
            self.build_pose_model(config.directory['pose_model'])
    
    def init_dir(self, config: Config) -> None:
        """ Initialize directories for storing raw video data and 2D keypoints data based on the provided config. """
        self.videos_raw_dir = Path(config.sub_directory['videos_raw'])
        self.points_2d_path = Path(config.sub_directory['points_2d']) / 'original.h5'
    
    def init_data_cfg(self, config: Config) -> None:
        """ Initialize necessary data for tracking and keypoints detection. """
        self.n_tracks = config.animal['n_tracks']
        self.n_bodyparts = len(config.animal['bodyparts'])
        self.label_mapping = config.animal['label_mapping']
        # The detector's classes are dye colours. Set dye_identity: false when the animals
        # are unmarked, or marked in a colour the detector was not trained on.
        self.dye_identity = config.animal.get('dye_identity', True)
        # Colour anchoring: pin the dyed animal to a fixed track slot from its coat colour.
        # Only used when dye_identity is False (i.e. the detector's classes are unusable).
        self.dye_color_identity = config.animal.get('dye_color_identity', False)
        self.dye_hue_range = tuple(config.animal.get('dye_hue_range', (140, 175)))  # OpenCV H, 0..179
        self.dye_sat_min = config.animal.get('dye_sat_min', 60)
        self.dye_val_min = config.animal.get('dye_val_min', 50)
        self.dye_min_pixels = config.animal.get('dye_min_pixels', 120)
        self.dye_margin = config.animal.get('dye_margin', 0.5)   # 2nd box must be < this x best
        self.dye_track = config.animal.get('dye_track', 1)       # slot the dyed animal owns
        self.skip_index = [config.animal['bodyparts'].index(bp) for bp in config.animal['skip']]

        self.bbox_threshold = config.threshold['bbox']
        self.iou_threshold = config.threshold['iou']
        self.keypoint_threshold = config.threshold['keypoint']
    
    def build_det_model(self, det_model_dir: str) -> None:
        """
        Build the detection model from the configuration file and checkpoint found in the given directory.

        Args:
            det_model_dir: The directory path where the detection model configuration and checkpoint are stored.
        """
        # Unique checkpoint with suffix .pth and 'best' in the name
        pth_checkpoint_files = list(Path(det_model_dir).glob('*best*.pth'))
        if len(pth_checkpoint_files) > 1:
            raise ValueError(f'Multiple best checkpoint files found in {det_model_dir}')
        elif len(pth_checkpoint_files) == 1:
            # Unique config with suffix .py
            config_files = list(Path(det_model_dir).glob('*.py'))
            assert len(config_files) == 1, f'Zero/Multiple config files found in {det_model_dir}'

            det_config = str(config_files[0])
            logger.info(f'Load detection model config from {det_config}')

            det_checkpoint = str(pth_checkpoint_files[0])
            logger.info(f'Load detection model checkpoint from {det_checkpoint}')

            self.detector = init_detector(det_config, det_checkpoint, device=self.device)
            self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        else:
            trt_checkpoint_files = list(Path(det_model_dir).glob('*best*.engine'))
            assert len(trt_checkpoint_files) == 1, f'Zero/Multiple best checkpoint files found in {det_model_dir}'

            det_configs = list(Path(det_model_dir).glob('*.py'))
            deploy_cfg = next((str(cfg) for cfg in det_configs if 'deploy_config' == str(cfg.stem)), None)
            assert deploy_cfg, f'No deploy config found in {det_model_dir}'
            model_cfg = next((str(cfg) for cfg in det_configs if 'model_config' == str(cfg.stem)), None)
            assert model_cfg, f'No model config found in {det_model_dir}'

            logger.warning(f'Using deployed model: {trt_checkpoint_files[0]} for detection, which has faster inference speed but lower accuracy.')

            assert HAS_MMDEPLOY, (
                f'Found only a TensorRT model in {det_model_dir}, which needs mmdeploy + TensorRT. '
                'Install mmdeploy, or place a *best*.pth checkpoint there to use the PyTorch model.')

            deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

            self.det_task_processor = build_task_processor(model_cfg, deploy_cfg, 'cuda')
            self.det_model_deployed = self.det_task_processor.build_backend_model([str(trt_checkpoint_files[0])])
            self.det_input_shape = get_input_shape(deploy_cfg)
            self.use_deployed_det = True

    def build_pose_model(self, pose_model_dir: str) -> None:
        """
        Build the pose estimation model from the configuration file and checkpoint found in the given directory.

        Args:
            pose_model_dir: The directory path where the pose estimation model configuration and checkpoint are stored.
        """
        # Unique checkpoint with suffix .pth and 'best' in the name
        checkpoint_files = list(Path(pose_model_dir).glob('*best*.pth'))
        if len(checkpoint_files) > 1:
            raise ValueError(f'Multiple best checkpoint files found in {pose_model_dir}')
        elif len(checkpoint_files) == 1:
            # Unique config with suffix .py
            config_files = list(Path(pose_model_dir).glob('*.py'))
            assert len(config_files) == 1, f'Zero/Multiple config files found in {pose_model_dir}'

            pose_config = str(config_files[0])
            logger.info(f'Load pose model config from {pose_config}')

            pose_checkpoint = str(checkpoint_files[0])
            logger.info(f'Load pose model checkpoint from {pose_checkpoint}')

            self.pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device=self.device)
        else: 
            trt_checkpoint_files = list(Path(pose_model_dir).glob('*best*.engine'))
            assert len(trt_checkpoint_files) == 1, f'Zero/Multiple best checkpoint files found in {pose_model_dir}'

            pose_configs = list(Path(pose_model_dir).glob('*.py'))
            deploy_cfg = next((str(cfg) for cfg in pose_configs if 'deploy_config' == str(cfg.stem)), None)
            assert deploy_cfg, f'No deploy config found in {pose_model_dir}'
            model_cfg = next((str(cfg) for cfg in pose_configs if 'model_config' == str(cfg.stem)), None)
            assert model_cfg, f'No model config found in {pose_model_dir}'
            data_meta_cfg = next((str(cfg) for cfg in pose_configs if 'data_meta' == str(cfg.stem)), None)
            assert data_meta_cfg, f'No data meta config found in {pose_model_dir}'

            logger.warning(f'Using deployed model: {trt_checkpoint_files[0]} for pose estimation, which has faster inference speed but lower accuracy.')

            assert HAS_MMDEPLOY, (
                f'Found only a TensorRT model in {pose_model_dir}, which needs mmdeploy + TensorRT. '
                'Install mmdeploy, or place a *best*.pth checkpoint there to use the PyTorch model.')

            deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

            task_processor = build_task_processor(model_cfg, deploy_cfg, 'cuda')
            self.pose_model_cfg = model_cfg
            self.pose_model_deployed = task_processor.build_backend_model([str(trt_checkpoint_files[0])])
            self.pose_dataset_meta = parse_pose_metainfo(dict(from_file=data_meta_cfg))
            self.use_deployed_pose = True
    
    def predict(self, vid_path: str = None) -> None:
        """ 
        Predict 2D poses for all videos in the raw video directory or the given video path. 
        
        Args:
            vid_path: The path to the video file to predict 2D poses for. If not provided, all videos in the raw video directory will be processed.
        """
        if vid_path:
            self.predict_video(vid_path)
        else:
            video_paths = sorted(self.videos_raw_dir.glob(f"*.mp4"))
            for video_path in video_paths:
                self.predict_video(str(video_path))
    
    def predict_video(self, video_path: str) -> None:
        """ 
        Predict 2D poses for the given video. 
        
        Args:
            video_path: The path to the video file to predict 2D poses for.
        """
        logger.info(f'Predicting 2D poses for: {video_path}')

        self._prev_centres = None  # track continuity is per-video, not across videos

        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        points_with_score_2d = []
        bboxes = []

        with trange(n_frames, ncols=100, desc='Predicting ... ', unit='frames') as progress_bar:
            frames_processed = 0
            while frames_processed < n_frames:
                batch_frames = self.prepare_batch_frames(cap, self.batch_size)
                if not batch_frames: break
                
                points_with_score_2d_batch, bboxes_batch = self.predict_image_batch(batch_frames)
                points_with_score_2d.extend(points_with_score_2d_batch)
                bboxes.extend(bboxes_batch)
                
                n_processed = len(batch_frames)
                frames_processed += n_processed
                progress_bar.update(n_processed)

        cap.release()

        points_with_score_2d = np.array(points_with_score_2d).transpose(1, 0, 2, 3)
        bboxes = np.array(bboxes).transpose(1, 0, 2)

        points_with_score_2d[:, :, self.skip_index] = np.nan
        save_points_bboxes_2d_h5(points=points_with_score_2d,
                                 bboxes=bboxes,
                                 name=Path(video_path).stem,
                                 file_path=self.points_2d_path)
    
    @staticmethod
    def prepare_batch_frames(cap, batch_size) -> List[np.ndarray]:
        """
        Prepare a batch of frames from the video capture object.
        
        Args:
            cap: The video capture object to read frames from.
            batch_size: The number of frames to read in a batch.
        
        Returns:
            A list of frames, where each frame is a numpy.ndarray representing an image.
        """
        batch_frames = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret: break
            batch_frames.append(frame)
        
        return batch_frames

    def predict_image_batch(self, imgs: List[np.ndarray]):
        """
        Predict keypoints and bounding boxes for a batch of images.

        Args:
            imgs: A list of images, where each image is represented as a numpy.ndarray.

        Returns:
            A tuple containing two lists:
            - points_with_score_2d_batch: A list of keypoints with scores for each image.
            - bboxes_batch: A list of bounding boxes for each image, with each bounding box described by its coordinates and score.
        """
        if not self.use_deployed_det:
            det_results = detector_predict(self.detector, imgs)
        else:
            det_results = detector_predict_deployed(self.det_model_deployed, self.det_task_processor, self.det_input_shape, imgs)

        bboxes_list = self.prepare_bbox(det_results, imgs) # (x1, y1, x2, y2, score, label)

        if not self.use_deployed_pose:
            pose_results_list = pose_topdown_predict(self.pose_estimator, imgs, bboxes_list)
        else:
            pose_results_list = pose_predict_deployed(self.pose_model_deployed, self.pose_model_cfg, self.pose_dataset_meta, imgs, bboxes_list)

        points_with_score_2d_batch, bboxes_batch = self.merge_pose_results_batch(bboxes_list, pose_results_list)

        return points_with_score_2d_batch, bboxes_batch
    
    def prepare_bbox(self, det_results: List[DetDataSample], imgs: List[np.ndarray] = None) -> List[np.ndarray]:
        """
        Process detection results to filter and assign bounding boxes based on scores and IoU thresholds.

        Args:
            det_results: A list of detection results, where each result corresponds to a frame and contains predicted instances.
            imgs: The frames the detections came from. Only needed for dye-colour identity anchoring.

        Returns:
            A list of numpy arrays, each containing the filtered and assigned bounding boxes for a frame. Each bounding box shape of (x1, y1, x2, y2, score, label).
        """
        bboxes_list = [] # Each element is a list of bboxes for a frame, could be variable length
        for frame_idx, det_result in enumerate(det_results):
            pred_instances = det_result.pred_instances.cpu().numpy()
            bboxes_all = np.concatenate((pred_instances.bboxes, pred_instances.scores[:, None], pred_instances.labels[:, None]), axis=1) # (x1, y1, x2, y2, score, label)

            bboxes_filtered = bboxes_all[pred_instances.scores > self.bbox_threshold]

            if self.label_mapping:
                bboxes_filtered = self.bboxes_label_mapping(bboxes_filtered, self.label_mapping) # Only keep the bboxes with the desired labels

            if self.dye_identity:
                bboxes = self.heuristic_assign_bboxes(bboxes_filtered, self.iou_threshold)
            else:
                bboxes = self.assign_bboxes_by_score(bboxes_filtered, self.n_tracks, self.iou_threshold)
                img = imgs[frame_idx] if imgs is not None and frame_idx < len(imgs) else None
                anchored = self._anchor_identity_by_dye(bboxes, img)
                if anchored is not None:
                    bboxes = anchored                      # absolute identity from the dye
                    self._remember_centres(bboxes)
                else:
                    bboxes = self._assign_track_ids_temporal(bboxes)   # coast on continuity

            bboxes_list.append(bboxes)
        return bboxes_list
    
    def _dye_pixels(self, img: np.ndarray, bbox: np.ndarray) -> int:
        """Count pixels inside a box whose colour matches the dye."""
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(img.shape[1], int(bbox[2])), min(img.shape[0], int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return 0
        hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lo, hi = self.dye_hue_range
        mask = (H >= lo) & (H <= hi) & (S >= self.dye_sat_min) & (V >= self.dye_val_min)
        return int(mask.sum())

    def _anchor_identity_by_dye(self, bboxes: np.ndarray, img: np.ndarray):
        """Pin the dyed animal to a fixed track slot using its coat colour.

        This is what makes identity ABSOLUTE rather than merely continuous: temporal
        matching keeps ids attached to animals but cannot recover after a full occlusion,
        so errors accumulate. Re-grounding on the dye whenever it is visible stops drift.

        Returns None when the dye is not clearly visible, or when two boxes look similarly
        dyed, so the caller can fall back to temporal continuity.
        """
        if not self.dye_color_identity or img is None or bboxes.size == 0 or self.n_tracks < 2:
            return None

        counts = np.array([self._dye_pixels(img, b) for b in bboxes], dtype=float)
        best = int(np.argmax(counts))
        if counts[best] < self.dye_min_pixels:
            return None                                  # no dye visible this frame
        rest = np.delete(counts, best)
        if rest.size and rest.max() > self.dye_margin * counts[best]:
            return None                                  # ambiguous between two boxes

        ids = np.full(len(bboxes), -1.0)
        ids[best] = self.dye_track
        free = [s for s in range(self.n_tracks) if s != self.dye_track]
        for i in range(len(ids)):
            if ids[i] < 0 and free:
                ids[i] = free.pop(0)
        out = bboxes.copy()
        out[:, 5] = ids
        return out[out[:, 5] >= 0]

    def _remember_centres(self, bboxes: np.ndarray) -> None:
        """Record per-slot box centres so temporal matching can resume after anchoring."""
        prev = getattr(self, '_prev_centres', None)
        new_prev = np.full((self.n_tracks, 2), np.nan)
        for b in bboxes:
            slot = int(b[5])
            if 0 <= slot < self.n_tracks:
                new_prev[slot] = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
        if prev is not None:
            missing = np.isnan(new_prev[:, 0])
            new_prev[missing] = prev[missing]
        self._prev_centres = new_prev

    def _assign_track_ids_temporal(self, bboxes: np.ndarray) -> np.ndarray:
        """Relabel boxes so a track id follows the same animal over time.

        assign_bboxes_by_score orders by detection score, which is not stable: two
        animals in the same frame often score within 0.1 of each other, so the ranking
        flips and the ids swap back and forth. Matching box centres to the previous
        frame instead keeps ids attached to animals.

        This gives continuity within one camera only. Making ids agree BETWEEN cameras
        still needs Reconstructor3D.triangulate(reassign_id=True).
        """
        if bboxes.size == 0:
            return bboxes

        centres = np.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2,
                            (bboxes[:, 1] + bboxes[:, 3]) / 2], axis=1)
        prev = getattr(self, '_prev_centres', None)

        if prev is None or np.all(np.isnan(prev)):
            ids = np.arange(len(bboxes), dtype=float)
        else:
            cost = np.linalg.norm(centres[:, None, :] - prev[None, :, :], axis=2)
            cost = np.where(np.isnan(cost), 1e6, cost)
            rows, cols = linear_sum_assignment(cost)
            ids = np.full(len(bboxes), -1.0)
            for r, c in zip(rows, cols):
                if cost[r, c] < 1e6:
                    ids[r] = c
            free = [s for s in range(self.n_tracks) if s not in ids]
            for i in range(len(ids)):
                if ids[i] < 0 and free:
                    ids[i] = free.pop(0)

        bboxes = bboxes.copy()
        bboxes[:, 5] = ids

        new_prev = np.full((self.n_tracks, 2), np.nan)
        for i, slot in enumerate(ids):
            if 0 <= slot < self.n_tracks:
                new_prev[int(slot)] = centres[i]
        if prev is not None:
            missing = np.isnan(new_prev[:, 0])          # keep last known position
            new_prev[missing] = prev[missing]
        self._prev_centres = new_prev

        return bboxes[bboxes[:, 5] >= 0]

    @staticmethod
    def assign_bboxes_by_score(bboxes: np.ndarray, n_tracks: int, iou_threshold: float = 0.8) -> np.ndarray:
        """Keep the top n_tracks boxes by score, suppressing overlaps, ignoring the class.

        The detector's classes encode dye colour, and heuristic_assign_bboxes keeps at
        most one box per class. That silently drops animals whenever two of them get the
        same label -- which is what happens when the animals are unmarked, or marked in a
        colour the model was never trained on. Here the label is replaced by a positional
        track index instead, so identity must come from elsewhere (cross-view grouping
        via reassign_id, or a temporal tracker).
        """
        if bboxes.size == 0:
            return np.empty((0, 6))

        selected = []
        for bbox in bboxes[np.argsort(-bboxes[:, 4])]:
            if any(Iou(bbox[:4], kept[:4]) > iou_threshold for kept in selected):
                continue
            selected.append(bbox.copy())
            if len(selected) == n_tracks:
                break

        out = np.array(selected)
        out[:, 5] = np.arange(len(out))  # positional track ids, not dye classes
        return out

    @staticmethod
    def bboxes_label_mapping(bboxes: np.ndarray, label_mapping: dict) -> np.ndarray:
        mapped_bboxes = []
        for bbox in bboxes:
            if bbox[5] in label_mapping:
                bbox[5] = label_mapping[bbox[5]]
                mapped_bboxes.append(bbox)
        
        return np.array(mapped_bboxes) if mapped_bboxes else np.empty((0, 6))
    
    @staticmethod
    def heuristic_assign_bboxes(bboxes: np.ndarray, iou_threshold: float = 0.8) -> np.ndarray:
        """
        Assign bounding boxes heuristicly based on scores and IoU threshold.

        Args:
            bboxes: A numpy array of bounding boxes with shape [N, 6], where N is the number of boxes, and each box shape of (x1, y1, x2, y2, score, label).
            iou_threshold: The IoU threshold for filtering overlapping boxes. Defaults to 0.8.

        Returns:
            A numpy array of the selected bounding boxes after applying the heuristic.
        """
        sorted_bboxes = bboxes[np.argsort(-bboxes[:, 4])]

        selected_bboxes = []
        selected_labels = set()

        while sorted_bboxes.size > 0:
            current_bbox = sorted_bboxes[0]
            sorted_bboxes = np.delete(sorted_bboxes, 0, axis=0) 

            if current_bbox[5] in selected_labels:
                continue

            selected_labels.add(current_bbox[5])
            selected_bboxes.append(current_bbox)

            remaining_bboxes = []
            for bbox in sorted_bboxes:
                if bbox[5] == current_bbox[5] or Iou(bbox[:4], current_bbox[:4]) > iou_threshold:
                    continue
                remaining_bboxes.append(bbox)

            sorted_bboxes = np.array(remaining_bboxes)

            if len(selected_labels) == len(np.unique(bboxes[:, 5])) or sorted_bboxes.size == 0:
                break

        return np.array(selected_bboxes) if selected_bboxes else np.empty((0, 6))
    
    def merge_pose_results_batch(self, bboxes_list: List[np.ndarray], pose_results_list: List[np.ndarray]):
        """
        Merge pose results with bounding boxes for a batch of frames, applying a keypoint score threshold.

        Args:
            bboxes_list: A list of bounding boxes for each frame, with each sub-list containing the bounding boxes for all detected instances.
            pose_results_list: A list of pose estimation results for each frame, with each sub-list containing the results for all detected instances.

        Returns:
            A tuple containing two numpy arrays:
            - points_with_score_2d_batch: An array shape of (batch_size, n_tracks, n_bodyparts, 3), last dimension is (x, y, score).
            - bboxes_batch: A array representing the bounding boxes for each frame, shape of (batch_size, n_tracks, 4), each box is (x1, y1, x2, y2).
        """
        points_with_score_2d_batch = []
        bboxes_batch = []
        for bboxes, pose_results in zip(bboxes_list, pose_results_list):
            points_with_score_2d_per_frame, bboxes_per_frame = self.merge_pose_results(pose_results, bboxes, self.n_tracks, self.n_bodyparts)
            points_with_score_2d_per_frame[points_with_score_2d_per_frame[..., 2] < self.keypoint_threshold] = np.nan

            points_with_score_2d_batch.append(points_with_score_2d_per_frame)
            bboxes_batch.append(bboxes_per_frame)
        
        return np.array(points_with_score_2d_batch), np.array(bboxes_batch)
    
    @staticmethod
    def merge_pose_results(pose_results, bboxes, n_tracks, n_bodyparts) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge pose results with bounding boxes for a single frame, structuring the data per track and body part.

        Args:
            pose_results: A list of pose estimation results for each detected person in the frame.
            bboxes: A list of bounding boxes corresponding to the pose results, with each box represented by its coordinates.
            n_tracks: The number of instances expected in the frame.
        
        Returns:
            A tuple containing two numpy arrays:
            - points_with_score_2d_per_frame: An array shape of (n_tracks, n_bodyparts, 3), last dimension is (x, y, score).
            - bboxes_per_frame: A array representing the bounding boxes for each track, shape of (n_tracks, 4), each box is (x1, y1, x2, y2).
        """
        points_with_score_2d_per_frame = np.full((n_tracks, n_bodyparts, 3), np.nan)
        bboxes_per_frame = np.full((n_tracks, 4), np.nan)

        for d, bbox in zip(pose_results, bboxes):
            idx = int(bbox[5])
            if idx >= n_tracks:
                continue
            points_with_score_2d_per_frame[idx][..., :2] = d.pred_instances.keypoints
            points_with_score_2d_per_frame[idx][..., 2] = d.pred_instances.keypoint_scores

            bboxes_per_frame[idx] = bbox[:4]
        
        return points_with_score_2d_per_frame, bboxes_per_frame


def detector_predict(model: nn.Module, imgs: List[np.ndarray]) -> List[DetDataSample]:
    """
    Perform detection on a batch of images using the given model.

    Args:
        model: The detection model.
        imgs: A list of images.

    Returns:
        A list of detection data samples.
    """
    test_pipeline = get_test_pipeline_cfg(model.cfg)
    if isinstance(imgs[0], np.ndarray):
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline)

    data_batch = defaultdict(list)
    for img in imgs:
        data_ = dict(img=img, img_id=0)
        data_ = test_pipeline(data_)

        data_batch['inputs'].append(data_['inputs'])
        data_batch['data_samples'].append(data_['data_samples'])

    with torch.no_grad():
        results = model.test_step(data_batch)

    return results


def pose_topdown_predict(model: nn.Module, imgs: List[np.ndarray], bboxes_list: List[np.ndarray] = None) -> List[PoseDataSample]:
    """
    Predict poses for instances in each image given their bounding boxes.

    Args:
        model: The pose estimation model.
        imgs: A list of images.
        bboxes_list: A list of bounding boxes for each image. If None, the function will not perform pose estimation.

    Returns:
        A list of pose data samples.
    """
    scope = model.cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)
    test_pipeline = Compose(get_test_pipeline_cfg(model.cfg))

    data_list = []
    for img, bboxes in zip(imgs, bboxes_list):
        for bbox in bboxes:
            data_info = dict(img=img)
            data_info['bbox'] = bbox[None, :4]  # shape (1, 4)
            data_info['bbox_score'] = bbox[None, 4]  # shape (1,)
            data_info.update(model.dataset_meta)
            data_list.append(test_pipeline(data_info))

    if data_list:
        data_batch = pseudo_collate(data_list)
        with torch.no_grad():
            results_batch = model.test_step(data_batch)
        results = reshape_pose_results(results_batch, bboxes_list)
    else:
        results = [[] for i in range(len(imgs))]

    return results


def reshape_pose_results(pose_results: List[PoseDataSample], bboxes_list: List[List[np.ndarray]]) -> List[List[PoseDataSample]]:
    """
    Reshapes pose prediction results to match the input images and bounding boxes structure.

    Args:
        pose_results: A list of pose results.
        bboxes_list: A list of lists, where each sublist contains bounding boxes for a single image.

    Returns:
        A list of lists, where each sublist contains pose results for detected instances in the corresponding input image.
    """
    pose_results_list = []
    pose_result_idx = 0
    for bboxes in bboxes_list:
        current_image_pose_results = []
        for _ in bboxes:
            current_image_pose_results.append(pose_results[pose_result_idx])
            pose_result_idx += 1

        pose_results_list.append(current_image_pose_results)

    return pose_results_list


def Iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: The first bounding box, represented as a numpy ndarray with the format [x1, y1, x2, y2].
        box2: The second bounding box, represented as a numpy ndarray with the same format.

    Returns:
        The Intersection over Union (IoU) value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    return intersection / union if union != 0 else 0


def detector_predict_deployed(model, task_processor, input_shape, imgs: List[np.ndarray]) -> List[DetDataSample]:
    # register_all_modules(True)
    init_default_scope('mmdet')
    data_batch, _ = task_processor.create_input(imgs, input_shape)

    with torch.no_grad():
        results = model.test_step(data_batch)
    
    return results

def pose_predict_deployed(model, model_cfg, dataset_meta, imgs, bboxes_list):
    init_default_scope('mmpose')
    test_pipeline = Compose(get_test_pipeline_cfg(model_cfg))

    data_list = []
    for img, bboxes in zip(imgs, bboxes_list):
        for bbox in bboxes:
            data_info = dict(img=img)
            data_info['bbox'] = bbox[None, :4]  # shape (1, 4)
            data_info['bbox_score'] = bbox[None, 4]  # shape (1,)
            data_info.update(dataset_meta)
            data_list.append(test_pipeline(data_info))

    if data_list:
        data_batch = pseudo_collate(data_list)
        with torch.no_grad():
            results_batch = model.test_step(data_batch)
    else: 
        results_batch = [[] for i in range(len(imgs))]
        
    results = reshape_pose_results(results_batch, bboxes_list)
    
    return results
