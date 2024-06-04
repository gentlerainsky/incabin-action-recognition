coco_keypoint_names = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

coco_keypoint_connections = [
    [1, 2],
    [1, 3],
    [2, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
    [6, 7],
    [6, 8],
    [6, 12],
    [7, 9],
    [7, 13],
    [8, 10],
    [9, 11],
    [12, 13],
    [12, 14],
    [13, 15],
    [14, 16],
    [15, 17]
]

bone_connections = coco_keypoint_connections[:-4]