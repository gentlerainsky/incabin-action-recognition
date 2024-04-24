import torch
import numpy as np


class Pose3DEstimator:
    def __init__(self, LitModel, lifter_saved_model):
        self.lifter_saved_model = lifter_saved_model
        lit_model = LitModel.load_from_checkpoint(self.lifter_saved_model)
        self.lit_model = lit_model.eval()

    def inference(self, batch):
        pose2d = batch['pose_2d']
        pose2d = pose2d.to(self.lit_model.device).to(self.lit_model.dtype)
        pose3d = self.lit_model(pose2d)
        return pose3d
