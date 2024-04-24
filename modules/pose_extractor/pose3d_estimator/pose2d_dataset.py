import torch

import json
import numpy as np
from modules.pose_extractor.pose3d_estimator.utils.normalization import (
    center_pose2d_to_neck,
    normalize_2d_pose_to_pose,
)

class Pose2DDataset:
    def __init__(self, frame_info_list:list[dict], pose2d_list:list[dict]):
        self.frame_info_list = frame_info_list
        self.pose2d_list = pose2d_list
        self.num_keypoints = 13
        self.samples = []
        self.preprocess()

    def filter_relevance_joint(self, pose_2d):
        return pose_2d[:self.num_keypoints]

    def preprocess(self):
        for item in self.pose2d_list:
            frame_index = item['frame_index']
            pose_2d = np.array(item['pose_2d'])
            pose_2d = self.filter_relevance_joint(pose_2d)
            pose_2d, root_2d = center_pose2d_to_neck(pose_2d)
            pose_2d, w, h = normalize_2d_pose_to_pose(pose_2d)
            item = {
                'frame_index': frame_index,
                'pose_2d': pose_2d,
                'root_2d': root_2d,
                'scale_factor': [w, h]
            }
            self.samples.append(item)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> dict:
        return self.samples[idx]
