import pytorch_lightning as pl
from modules.action_recognizer.models.action_transformer.action_transformer import ActionTransformer
from torch import optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


class LitActionTransformer(pl.LightningModule):

    def __init__(self, model_kwargs, lr, is_pose_3d=False):
        super().__init__()
        self.save_hyperparameters()
        if is_pose_3d:
            model_kwargs['num_channels'] = 3
        else:
            model_kwargs['num_channels'] = 2

        self.model = ActionTransformer(**model_kwargs)
        self.val_predict = []
        self.val_gt = []
        self.test_predict = []
        self.test_gt = []
        self.is_pose_3d = is_pose_3d
        self.test_confusion_matrix = None
        # self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        activities = batch['activity']
        if self.is_pose_3d:
            pose = batch['pose_3d'].float()
        else:
            pose = batch['pose_2d'].float()
        # pose_3d = batch['pose_3d'].float()
        # valid_len = batch['valid_len']
        num_batches, num_frames, num_joints, num_channels = pose.shape
        x = pose.reshape([num_batches, num_frames, -1])
        
        preds = self.model(x)
        loss = F.cross_entropy(preds, activities)
        acc = (preds.argmax(dim=-1) == activities).float().mean()

        self.log(f'train_loss', loss)
        self.log(f'train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        activities = batch['activity']
        if self.is_pose_3d:
            pose = batch['pose_3d'].float()
        else:
            pose = batch['pose_2d'].float()
        num_batches, num_frames, num_joints, num_channels = pose.shape
        x = pose.reshape([num_batches, num_frames, -1])

        preds = self.model(x)
        predict = preds.argmax(dim=-1)
        self.val_predict.append(predict.detach().cpu().numpy())
        gt = activities
        self.val_gt.append(gt.detach().cpu().numpy())
        acc = (predict==gt).float().mean()
        return acc

    def test_step(self, batch, batch_idx):
        activities = batch['activity']
        if self.is_pose_3d:
            pose = batch['pose_3d'].float()
        else:
            pose = batch['pose_2d'].float()
        num_batches, num_frames, num_joints, num_channels = pose.shape
        x = pose.reshape([num_batches, num_frames, -1])
        
        preds = self.model(x)
        predict = preds.argmax(dim=-1)
        self.test_predict.append(predict.detach().cpu().numpy())
        gt = activities
        self.test_gt.append(gt.detach().cpu().numpy())
        acc = (predict==gt).float().mean()
        self.log(f'test_acc', acc)
        return acc

    def on_validation_epoch_end(self):
        predict = np.concatenate(self.val_predict)
        gt = np.concatenate(self.val_gt)
        accuracy = (predict == gt).mean()
        num_correct = (predict == gt).sum()
        print(f'validation set accuracy = {accuracy:.4f} [correct={num_correct}/{gt.shape[0]}]')
        # with np.printoptions(threshold=np.inf, linewidth=1000):
        #     matrix = confusion_matrix(gt, predict)
        #      print(f'validation set confusion_matrix =\n{matrix}')
        self.val_predict = []
        self.val_gt = []
        self.log(f'val_acc', accuracy)

    def on_test_epoch_end(self):
        predict = np.concatenate(self.test_predict)
        gt = np.concatenate(self.test_gt)
        matrix = confusion_matrix(gt, predict)
        self.test_confusion_matrix = matrix
        accuracy = (predict == gt).mean()
        num_correct = (predict == gt).sum()
        print(f'Test set accuracy = {accuracy:.4f} [correct={num_correct}/{gt.shape[0]}]')
        with np.printoptions(threshold=np.inf, linewidth=1000):
            print(f'Test set confusion_matrix =\n{matrix}')
        self.test_predict = []
        self.test_gt = []
        self.log(f'test_acc', accuracy)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]
