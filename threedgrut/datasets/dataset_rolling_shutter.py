import os
import numpy as np
import torch
from threedgrut.datasets.dataset_fisheye import FisheyeDataset
from threedgrut.datasets.camera_models import ShutterType
from threedgrut.utils.logger import logger
from collections import namedtuple
from scipy.spatial.transform import Rotation

class RollingShutterFisheyeDataset(FisheyeDataset):
    """
    Fisheye Dataset extending FisheyeDataset with rolling shutter support.  
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rolling_shutter_poses = self._load_rolling_shutter_poses()
        if self._rolling_shutter_poses:
            logger.info(f"🎉 Rolling shutter enabled: {len(self._rolling_shutter_poses)} poses")
            # self._print_rolling_shutter_summary()
            self._rebuild_intrinsics_with_rolling_shutter()
        else:
            logger.info("ℹ️  Rolling shutter disabled - no pose data found")

    def _rebuild_intrinsics_with_rolling_shutter(self):
        """rebuild intrinsics to set rolling shutter type"""
        new_intrinsics = {}
        
        for camera_id, (params_dict, rays_ori, rays_dir, camera_name) in self.intrinsics.items():
            # copy
            new_params_dict = params_dict.copy()
            new_params_dict["shutter_type"] = ShutterType.ROLLING_TOP_TO_BOTTOM
            
            new_intrinsics[camera_id] = (new_params_dict, rays_ori, rays_dir, camera_name)
            
        # replace
        self.intrinsics = new_intrinsics
        logger.info("📝 Updated camera intrinsics with ROLLING_TOP_TO_BOTTOM shutter type")

    def _print_rolling_shutter_summary(self):
        """print summary statistics of rolling shutter poses"""
        if not self._rolling_shutter_poses:
            return
            
        translation_diffs = []
        rotation_diffs = []
        
        for qvec_start, tvec_start, qvec_end, tvec_end in self._rolling_shutter_poses.values():
            t_diff = np.linalg.norm(tvec_end - tvec_start)
            translation_diffs.append(t_diff)
            
            dot = abs(np.dot(qvec_start, qvec_end))
            dot = np.clip(dot, 0, 1)
            angle_diff = 2 * np.arccos(dot) * 180 / np.pi
            rotation_diffs.append(angle_diff)
        
        logger.info("📊 Rolling Shutter Statistics:")
        logger.info(f"   Translation diff: min={np.min(translation_diffs):.6f}, "
                   f"max={np.max(translation_diffs):.6f}, "
                   f"mean={np.mean(translation_diffs):.6f}")
        logger.info(f"   Rotation diff: min={np.min(rotation_diffs):.2f}°, "
                   f"max={np.max(rotation_diffs):.2f}°, "
                   f"mean={np.mean(rotation_diffs):.2f}°")

    def _load_rolling_shutter_poses(self):
        """
        Read rolling shutter poses from images_ref_rs.txt file.
        
        File format (starting from line 7):
        IMAGE_ID QW_START QX_START QY_START QZ_START TX_START TY_START TZ_START 
                QW_END QX_END QY_END QZ_END TX_END TY_END TZ_END 1 IMAGE_NAME
        
        Returns:
            Dict mapping image names to rolling shutter pose data
        """
        rs_file = os.path.join(self.path, "colmap", "images_ref_rs.txt")
        if not os.path.exists(rs_file):
            logger.info("ℹ️  No rolling shutter file found - using global shutter mode")
            return {}
        
        logger.info(f"📖 Reading rolling shutter poses from: {rs_file}")    
        poses = {}
        
        with open(rs_file, 'r') as f:
            lines = f.readlines()
            # from line 7 onwards are data
            for line in lines[6:]:  
                line = line.strip()
                if not line or line.startswith('#'):
                    continue                    
                parts = line.split()
                if len(parts) >= 17:  # garentee enough parts
                    try:
                        image_name = parts[16]
                        qvec_start = np.array([float(x) for x in parts[1:5]])
                        tvec_start = np.array([float(x) for x in parts[5:8]])
                        qvec_end = np.array([float(x) for x in parts[8:12]])
                        tvec_end = np.array([float(x) for x in parts[12:15]])
                        poses[image_name] = (qvec_start, tvec_start, qvec_end, tvec_end)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Jump lines: {line[:50]}... 错误: {e}")
                        continue
        
        logger.info(f"📊 Loaded {len(poses)} rolling shutter poses")
        return poses

    def get_gpu_batch_with_intrinsics(self, batch):
        """rolling shutter poses"""
        gpu_batch = super().get_gpu_batch_with_intrinsics(batch)
        
        SensorPose3D = namedtuple('SensorPose3D', ['T_world_sensors', 'timestamps_us'])
        
        if (hasattr(self, 'current_image_name') and 
            self.current_image_name in self._rolling_shutter_poses):
            
            qvec_start, tvec_start, qvec_end, tvec_end = self._rolling_shutter_poses[self.current_image_name]
            
            #  [tx,ty,tz, qx,qy,qz,qw] 
            # COLMAP[qw,qx,qy,qz]，expect[qx,qy,qz,qw]
            start_pose = np.concatenate([tvec_start, qvec_start[1:], [qvec_start[0]]])
            end_pose = np.concatenate([tvec_end, qvec_end[1:], [qvec_end[0]]])
            
            # add to batch
            gpu_batch.sensor_poses = SensorPose3D(
                T_world_sensors=[
                    torch.tensor(start_pose, dtype=torch.float32, device=self.device),
                    torch.tensor(end_pose, dtype=torch.float32, device=self.device)
                ],
                timestamps_us=[0, 1]  # anything non-equal
            )
            
            # logging for first few images
            if not hasattr(self, '_rs_logged_count'):
                self._rs_logged_count = 0
            if self._rs_logged_count < 3:
                logger.info(f"✅ Rolling shutter poses added for {self.current_image_name}")
                logger.info(f"   Start: t={tvec_start} q={qvec_start}")
                logger.info(f"   End:   t={tvec_end} q={qvec_end}")
                logger.info(f"   Δt={np.linalg.norm(tvec_end - tvec_start):.6f}")
                self._rs_logged_count += 1
            elif self._rs_logged_count == 3:
                logger.info("   ... (rolling shutter continues for remaining images)")
                self._rs_logged_count += 1
        else:
            # Global shutter
            T_world = gpu_batch.T_to_world.squeeze().cpu().numpy()  # [4,4] matrix
            

            R = T_world[:3, :3]  
            t = T_world[:3, 3]   
            
   
            
            quat = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]
            
            # 7D pose: [tx,ty,tz, qx,qy,qz,qw]
            pose_7d = np.concatenate([t, quat])
            pose_tensor = torch.tensor(pose_7d, dtype=torch.float32, device=self.device)
            
            gpu_batch.sensor_poses = SensorPose3D(
                T_world_sensors=[pose_tensor.clone(), pose_tensor.clone()],  # 相同pose
                timestamps_us=[0, 1]
            )
            
        return gpu_batch

    def __getitem__(self, idx):
        self.current_image_name = os.path.basename(self.image_paths[idx])
        return super().__getitem__(idx)