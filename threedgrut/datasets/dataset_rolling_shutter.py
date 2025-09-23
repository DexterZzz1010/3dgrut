import os
import numpy as np
from threedgrut.datasets.dataset_fisheye import FisheyeDataset
from threedgrut.datasets.camera_models import ShutterType
from threedgrut.utils.logger import logger


class RollingShutterFisheyeDataset(FisheyeDataset):
    """
    Fisheye Dataset extending FisheyeDataset with rolling shutter support.  
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rolling_shutter_poses = self._load_rolling_shutter_poses()
        if self._rolling_shutter_poses:
            logger.info(f"Rolling shutter enabled: {len(self._rolling_shutter_poses)} poses")

    def _load_rolling_shutter_poses(self):
        """
        Read rolling shutter poses from images_ref_rs.txt file.
        
        File format:
        IMAGE_ID QW_START QX_START QY_START QZ_START TX_START TY_START TZ_START 
                QW_END QX_END QY_END QZ_END TX_END TY_END TZ_END 1 IMAGE_NAME
        
        Returns:
            Dict mapping image names to rolling shutter pose data
        """
        rs_file = os.path.join(self.path, "colmap", "images_ref_rs.txt")
        if not os.path.exists(rs_file):
            return {}
            
        poses = {}
        with open(rs_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) >= 16:
                    try:
                        image_name = parts[16]
                        qvec_start = np.array([float(x) for x in parts[1:5]])
                        tvec_start = np.array([float(x) for x in parts[5:8]])
                        qvec_end = np.array([float(x) for x in parts[8:12]])
                        tvec_end = np.array([float(x) for x in parts[12:15]])
                        poses[image_name] = (qvec_start, tvec_start, qvec_end, tvec_end)
                    except:
                        continue
        return poses

    def _determine_shutter_type_for_image(self, image_name):
        """确定快门类型 - 唯一需要override的核心逻辑"""
        return (ShutterType.ROLLING_TOP_TO_BOTTOM 
                if image_name in self._rolling_shutter_poses 
                else ShutterType.GLOBAL)

    def get_gpu_batch_with_intrinsics(self, batch):
        """添加rolling shutter信息到batch"""
        gpu_batch = super().get_gpu_batch_with_intrinsics(batch)
        
        if hasattr(self, 'current_image_name') and self.current_image_name in self._rolling_shutter_poses:
            qvec_start, tvec_start, qvec_end, tvec_end = self._rolling_shutter_poses[self.current_image_name]
            # [tx,ty,tz,qx,qy,qz,qw]
            start_pose = np.concatenate([tvec_start, qvec_start[1:], [qvec_start[0]]])
            end_pose = np.concatenate([tvec_end, qvec_end[1:], [qvec_end[0]]])
            
            
            gpu_batch.rolling_shutter_start_pose = torch.tensor(start_pose, device=self.device)
            gpu_batch.rolling_shutter_end_pose = torch.tensor(end_pose, device=self.device)
            gpu_batch.has_rolling_shutter = True
        else:
            gpu_batch.has_rolling_shutter = False
            
        return gpu_batch

    def __getitem__(self, idx):
        """记录当前图像名称"""
        self.current_image_name = os.path.basename(self.image_paths[idx])
        return super().__getitem__(idx)