import numpy as np
from torch.nn.utils.rnn import pad_sequence

all_activities = [
    'closing_bottle',
    'closing_door_inside',
    'closing_door_outside',
    'closing_laptop',
    'drinking',
    'eating',
    'entering_car',
    'exiting_car',
    'fastening_seat_belt',
    'fetching_an_object',
    'interacting_with_phone',
    'looking_or_moving_around (e.g. searching)',
    'opening_backpack',
    'opening_bottle',
    'opening_door_inside',
    'opening_door_outside',
    'opening_laptop',
    'placing_an_object',
    'preparing_food',
    'pressing_automation_button',
    'putting_laptop_into_backpack',
    'putting_on_jacket',
    'putting_on_sunglasses',
    'reading_magazine',
    'reading_newspaper',
    'sitting_still',
    'taking_laptop_from_backpack',
    'taking_off_jacket',
    'taking_off_sunglasses',
    'talking_on_phone',
    'unfastening_seat_belt',
    'using_multimedia_display',
    'working_on_laptop',
    'writing'
]
all_activity_mapper = {all_activities[i]: i for i in range(len(all_activities))}


class VideoDataset:
    def __init__(self, annotation_df, poses, max_len=30) -> None:
        self.pose_info = poses
        self.annotation_df = annotation_df
        self.max_len = max_len
        self.all_activities = set([])
        self.pose_2d = [item['pose_2d'] for item in self.pose_info]
        self.pose_3d = [item['pose_3d'] for item in self.pose_info]
        self.shuffle()

    def shuffle(self):
        self.pose_2d_sequences = []
        self.pose_3d_sequences = []
        self.sequnce_valid_len = []
        self.activities = []
        for idx, annotation in self.annotation_df.iterrows():
            start, end = annotation.frame_index_start, annotation.frame_index_end
            seq_start = start
            valid_len = self.max_len
            if (end - start) > self.max_len:
                seq_start = np.random.randint(start, end - self.max_len)
                pose_2d = np.array(self.pose_2d[seq_start: seq_start + self.max_len])
                pose_3d = np.array(self.pose_3d[seq_start: seq_start + self.max_len])
            else:
                pose_2d = np.array(self.pose_2d[seq_start: end])
                pose_3d = np.array(self.pose_3d[seq_start: end])
                if (end - start) < self.max_len:
                    dim0, dim1, dim2 = pose_2d.shape
                    pose_2d = np.concatenate([np.array(pose_2d), np.zeros((self.max_len - dim0, dim1, dim2))], axis=0)
                    dim0, dim1, dim2 = pose_3d.shape
                    pose_3d = np.concatenate([np.array(pose_3d), np.zeros((self.max_len - dim0, dim1, dim2))], axis=0)
                    valid_len = self.max_len - dim0
            self.pose_2d_sequences.append(pose_2d)
            self.pose_3d_sequences.append(pose_3d)
            self.sequnce_valid_len.append(valid_len)
            self.activities.append(all_activity_mapper[annotation.activity])

    def __len__(self):
        return self.annotation_df.shape[0]

    def __getitem__(self, idx):
        return dict(
            idx=idx,
            activity=self.activities[idx],
            pose_2d=self.pose_2d_sequences[idx],
            pose_3d=self.pose_3d_sequences[idx],
            valid_len=self.sequnce_valid_len[idx]
        )
