import numpy as np
import pandas as pd


all_activities = sorted([
    'standing_by_the_door', 'closing_door_outside',
    'opening_door_outside', 'entering_car', 'closing_door_inside',
    'fastening_seat_belt', 'using_multimedia_display', 'sitting_still',
    'pressing_automation_button', 'fetching_an_object',
    'opening_laptop', 'working_on_laptop', 'interacting_with_phone',
    'closing_laptop', 'placing_an_object', 'unfastening_seat_belt',
    'putting_on_jacket', 'opening_bottle', 'drinking',
    'closing_bottle', 'looking_or_moving_around (e.g. searching)',
    'preparing_food', 'eating', 'looking_back_left_shoulder',
    'taking_off_sunglasses', 'putting_on_sunglasses',
    'reading_newspaper', 'writing', 'talking_on_phone',
    'reading_magazine', 'taking_off_jacket', 'opening_door_inside',
    'exiting_car', 'opening_backpack', 'closing_backpack',
    'putting_laptop_into_backpack', 'looking_back_right_shoulder',
    'taking_laptop_from_backpack', 'moving_towards_door'
])
all_activity_mapper = {all_activities[i]: i for i in range(len(all_activities))}


class PoseDataset:
    def __init__(self, annotation_df, pose_df, max_len=30, features=None):
        self.pose_df = pose_df
        self.annotation_df = annotation_df
        self.max_len = max_len
        self.items = []
        if features is None:
            self.features = set(['pose_2d'])
        else:
            self.features = set(features)
        self.calculate_features()

    def calculate_features(self):
        for index, row in self.annotation_df.iterrows():
            start, end = row.frame_index_start, row.frame_index_end
            frame_seq = self.pose_df.loc[start: end - 1]
            if frame_seq[frame_seq.annotated_frame].shape[0] > self.max_len:
                frame_seq = frame_seq[frame_seq.annotated_frame]
            
            item = dict(
                index=index,
                seq_len=frame_seq.shape[0],
                pose_2d=np.stack(frame_seq['pose_2d'].values, axis=0),
                pose_3d=np.stack(frame_seq['pose_3d'].values, axis=0),
                bone=np.stack(frame_seq['bone'].values, axis=0),
                velocity=np.stack(frame_seq['velocity'].values, axis=0),
                activity=all_activity_mapper[row.activity]
            )
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        seq_start = np.random.randint(0, item['seq_len'] - self.max_len)
        results = dict(
            idx=idx,
            activity=item['activity'],
            pose_2d=item['pose_2d'][seq_start:seq_start+self.max_len],
            pose_3d=item['pose_3d'][seq_start:seq_start+self.max_len],
            bone=item['bone'][seq_start:seq_start+self.max_len],
            velocity=item['velocity'][seq_start:seq_start+self.max_len],
            valid_len=self.max_len
        )
        return results

