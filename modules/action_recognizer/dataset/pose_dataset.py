import numpy as np
from torch.nn.utils.rnn import pad_sequence

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
    def __init__(self, annotation_df, poses, max_len=30) -> None:
        self.pose_info = poses
        self.annotation_df = annotation_df
        self.max_len = max_len
        self.pose_2d = np.array([item['pose_2d'] for item in self.pose_info])
        self.pose_3d = np.array([item['pose_3d'] for item in self.pose_info])

    def __len__(self):
        return self.annotation_df.shape[0]

    def __getitem__(self, idx):
        item = self.annotation_df.iloc[idx]
        seq_start = np.random.randint(item.frame_index_startme_, item.frame_index_end - self.max_len)
        return dict(
            idx=idx,
            activity=all_activity_mapper[item.activity],
            pose_2d=self.pose_2d[seq_start: seq_start + self.max_len],
            pose_3d=self.pose_3d[seq_start: seq_start + self.max_len],
            valid_len=self.max_len
        )
