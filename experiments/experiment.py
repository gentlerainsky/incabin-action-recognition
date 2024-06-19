import sys
# sys.path.append('/workspace')
sys.path.append('.')
import pickle
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from modules.action_recognizer.dataset.pose_dataset import PoseDataset
from experiments.action_transformer_experiment_config import experiment_config as action_transformer_experiment_config
from experiments.sttformer_experiment_config import experiment_config as sttformer_experiment_config
from experiments.pose_transformer_experiment_config import experiment_config as pose_transformer_experiment_config
from experiments.stgcn_experiment_config import experiment_config as stgcn_experiment_config

import shutil
import json


pl.seed_everything(1234)
with open('output/inner_mirror_with_padding/pose_info_with_bone_velo.pkl', 'rb') as f:
    pose_df = pickle.load(f)

with open('output/inner_mirror_with_padding/annotation.pkl', 'rb') as f:
    annotation_df = pickle.load(f)

# split 0
# train 1,2,3,4,6,7,8,9,10,12
# val 14, 15
# test 5, 11, 13
train_participants = [1,2,3,4,6,7,8,9,10,12]
val_participants = [14, 15]
test_participants = [5, 11, 13]
train_data = annotation_df[annotation_df.participant_id.isin(train_participants)]
val_data = annotation_df[annotation_df.participant_id.isin(val_participants)]
test_data = annotation_df[annotation_df.participant_id.isin(test_participants)]

train_dataset = PoseDataset(train_data, pose_df, max_len=30)
val_dataset = PoseDataset(val_data, pose_df, max_len=30)
test_dataset = PoseDataset(test_data, pose_df, max_len=30)

num_workers = 16
class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=num_workers)

experiment_config = (
    action_transformer_experiment_config
    + sttformer_experiment_config
    + pose_transformer_experiment_config
    + stgcn_experiment_config
)


for experiment in experiment_config:
    experiment_name = experiment['name']
    lit_model = experiment['lit_model']
    saved_model_path = experiment['saved_path']
    batch_size = experiment.get('batch_size', 256)
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_acc', mode='max', save_top_k=1
    )
    early_stopping = EarlyStopping(
        monitor='val_acc',  mode="max", patience=3
    )
    dm = DataModule(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_epoch = 300
    saved_model_path = f'saved_models/action_recognizer/{saved_model_path}/{experiment_name}'
    if Path(saved_model_path).exists():
        shutil.rmtree(saved_model_path)

    enable_progress_bar = True
    num_sanity_val_steps = 10
    val_check_period = 10

    trainer = pl.Trainer(
        # max_steps=10,
        max_epochs=max_epoch,
        callbacks=[
            model_checkpoint_callback,
            early_stopping
        ],
        accelerator=device,
        check_val_every_n_epoch=val_check_period,
        default_root_dir=saved_model_path,
        gradient_clip_val=1.0,
        logger=enable_progress_bar,
        enable_progress_bar=enable_progress_bar,
        num_sanity_val_steps=num_sanity_val_steps,
        log_every_n_steps=5,
        reload_dataloaders_every_n_epochs=1
    )

    trainer.fit(lit_model, datamodule=dm)
    trainer.test(lit_model, datamodule=dm)
    test_matrix = lit_model.test_confusion_matrix.astype(float)
    test_accuracy = lit_model.test_accuracy
    with open(f'{saved_model_path}/result.json', 'w') as f:
        f.write(json.dumps(dict(
            test_accuracy = test_accuracy,
            test_confusion_matrix = test_matrix.tolist()
        ), indent=2))
