import numpy as np
import torch

class GeoDatasetMixin:
    """
    A mixin class containing shared logic for GeoProp datasets (S3DIS/ScanNet).
    It handles:
    1. Training Data Augmentation (Rotation, Jitter, Color) - Aligned with PTv3
    2. Test-Time Augmentation (TTA) - Combinatorial Strategy
    3. Sliding Window Validation Logic
    """

    def apply_training_augmentation(self, coord, color, rot_z_range=[-1, 1], tilt_range=[-1/64, 1/64]):
        """
        Apply training augmentations aligned with official Pointcept PTv3 config.
        Includes:
        - Z-axis Rotation (Large)
        - X/Y-axis Tilt (Small)
        - Scaling
        - Flipping
        - Random Jitter (Gaussian Noise)
        - Chromatic Augmentation (Jitter, AutoContrast, Translation)
        - Color Drop (GeoProp specific)
        """
        
        # 1. Z-Axis Rotation: [-1, 1] * pi (i.e., -180 to 180 degrees) with p=0.5
        if np.random.random() < 0.5:
            angle_z = np.random.uniform(rot_z_range[0], rot_z_range[1]) * np.pi
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
            R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], dtype=np.float32)
            coord = np.dot(coord, R_z.T)

        # 2. X-Axis Tilt: approx +/- 2.8 degrees with p=0.5
        if np.random.random() < 0.5:
            angle_x = np.random.uniform(tilt_range[0], tilt_range[1]) * np.pi
            cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
            R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=np.float32)
            coord = np.dot(coord, R_x.T)

        # 3. Y-Axis Tilt: approx +/- 2.8 degrees with p=0.5
        if np.random.random() < 0.5:
            angle_y = np.random.uniform(tilt_range[0], tilt_range[1]) * np.pi
            cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
            R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)
            coord = np.dot(coord, R_y.T)
        
        # 4. Random Scale: [0.9, 1.1]
        scale = np.random.uniform(scale_range[0], scale_range[1])
        coord *= scale
        
        # 5. Random Flip (X-axis)
        if np.random.random() > 0.5: 
            coord[:, 0] = -coord[:, 0]
        
        # 6. Random Jitter (Gaussian Noise) - Critical for Regularization
        # Aligned with PTv3 official config: sigma=0.005, clip=0.02
        noise = np.clip(jitter_sigma * np.random.randn(coord.shape[0], 3), -jitter_clip, jitter_clip)
        coord += noise

        # 7. Chromatic Augmentation
        # 7.1 AutoContrast (Simple simulation)
        if np.random.random() < 0.2:
            c_min, c_max = np.min(color, axis=0), np.max(color, axis=0)
            scale = 255.0 / (c_max - c_min + 1e-6)
            color = (color - c_min) * scale

        # 7.2 Chromatic Translation & Jitter (p=0.95 in official, we use conservative settings)
        if np.random.random() < 0.95: 
            # Translation
            tr_ratio = 0.05
            tr = np.random.uniform(-tr_ratio, tr_ratio, 3) * 255
            color += tr
            # Jitter
            noise = np.random.randn(1).astype(np.float32)
            color = color * (1 + 0.05 * noise)

        # 8. Color Drop (GeoProp specific strategy for weak supervision)
        if np.random.random() < color_drop_prob:
            color[:] = 0.0

        return coord, color

    def apply_training_augmentation(self, coord, color, 
                                    rot_z_range=[-1, 1], 
                                    tilt_range=[-1/64, 1/64],
                                    scale_range=[0.9, 1.1], 
                                    jitter_sigma=0.005, 
                                    jitter_clip=0.02,
                                    color_drop_prob=0.2):
        """
        Generates sliding window crops for validation/testing.
        Supports combinatorial TTA (aligned with official PTv3).
        """
        fragment_list = []
        coord_min = np.min(coord, axis=0)
        coord_max = np.max(coord, axis=0)
        visited_mask = np.zeros(coord.shape[0], dtype=bool)
        
        stride_x, stride_y = stride, stride
        
        grid_x = np.arange(coord_min[0], coord_max[0] + stride_x, stride_x)
        grid_y = np.arange(coord_min[1], coord_max[1] + stride_y, stride_y)
        
        if scan_mode == 'xyz':
            stride_z = stride
            grid_z = np.arange(coord_min[2], coord_max[2] + stride_z, stride_z)
        else:
            z_center = (coord_min[2] + coord_max[2]) / 2.0
            grid_z = [z_center] 
        
        # Construct TTA list (Combinatorial Strategy)
        transforms_to_apply = []
        if tta_conf.get('enable'):
            # 1. Base 4 Rotations
            transforms_to_apply.extend([
                dict(scale=1.0, rot_z=0, flip_x=False, flip_y=False),
                dict(scale=1.0, rot_z=1, flip_x=False, flip_y=False),
                dict(scale=1.0, rot_z=2, flip_x=False, flip_y=False),
                dict(scale=1.0, rot_z=3, flip_x=False, flip_y=False),
            ])
            # 2. Rotations + Scale 0.95
            transforms_to_apply.extend([
                dict(scale=0.95, rot_z=0, flip_x=False, flip_y=False),
                dict(scale=0.95, rot_z=1, flip_x=False, flip_y=False),
                dict(scale=0.95, rot_z=2, flip_x=False, flip_y=False),
                dict(scale=0.95, rot_z=3, flip_x=False, flip_y=False),
            ])
            # 3. Rotations + Scale 1.05
            transforms_to_apply.extend([
                dict(scale=1.05, rot_z=0, flip_x=False, flip_y=False),
                dict(scale=1.05, rot_z=1, flip_x=False, flip_y=False),
                dict(scale=1.05, rot_z=2, flip_x=False, flip_y=False),
                dict(scale=1.05, rot_z=3, flip_x=False, flip_y=False),
            ])
            # 4. Flip
            transforms_to_apply.append(dict(scale=1.0, rot_z=0, flip_x=True, flip_y=False))
        else:
            transforms_to_apply = [dict(scale=1.0, rot_z=0, flip_x=False, flip_y=False)]

        for x in grid_x:
            for y in grid_y:
                for z in grid_z:
                    center = np.array([x, y, z])
                    indices = self.get_knn_indices(coord, center=center, num_points=num_points)
                    visited_mask[indices] = True
                    
                    coord_chunk = coord[indices]
                    color_chunk = color[indices]
                    segment_chunk = segment[indices]

                    for t_cfg in transforms_to_apply:
                        coord_aug = coord_chunk.copy()
                        
                        # Apply Combinatorial TTA
                        rot_k = t_cfg.get('rot_z', 0) 
                        if rot_k > 0:
                            angle = rot_k * np.pi / 2
                            c, s = np.cos(angle), np.sin(angle)
                            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                            coord_aug = coord_aug @ R.T
                        
                        scale_val = t_cfg.get('scale', 1.0) 
                        if scale_val != 1.0:
                            coord_aug *= scale_val
                            
                        if t_cfg.get('flip_x', False): 
                            coord_aug[:, 0] = -coord_aug[:, 0]
                        
                        chunk_dict = self.prepare_input_dict(
                            coord_aug,      
                            color_chunk, 
                            segment_chunk, 
                            indices,        
                            is_test_fragment=True
                        )
                        fragment_list.append(chunk_dict)

        uncovered_count = np.sum(~visited_mask)
        return fragment_list, uncovered_count

    def get_knn_indices(self, coord, center=None, num_points=None):
        """
        Selects 'num_points' neighbors around a center point.
        """
        N = coord.shape[0]
        # Use instance num_points if not provided
        target_N = num_points if num_points is not None else self.num_points
        
        if center is None:
            # Random center for training
            center_idx = np.random.choice(N)
            center_point = coord[center_idx]
        else:
            # Specific center for sliding window
            center_point = center

        dist = np.sum((coord - center_point)**2, axis=1)
        
        if N < target_N:
            base = np.arange(N)
            pad = np.random.choice(N, target_N - N, replace=True)
            indices = np.concatenate([base, pad])
        else:
            indices = np.argpartition(dist, target_N)[:target_N]
            
        np.random.shuffle(indices)
        return indices