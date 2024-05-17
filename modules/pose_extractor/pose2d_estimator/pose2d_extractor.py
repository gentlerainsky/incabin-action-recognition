import cv2
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from modules.pose_extractor.pose2d_estimator.pose2d_estimator import Pose2DEstimator
import datetime


class Pose2DExtractor:
    def __init__(
        self,
        video_root_path: str,
        data_subset: str,
        pose_estimator_2d: Pose2DEstimator,
        pickle_output_path: str,
        padding_length: int = 15,
        sampling_period: int = 2
    ):
        self.video_root_path = video_root_path
        self.data_subset = data_subset
        self.pose_estimator_2d = pose_estimator_2d
        self.annotation_df = None
        self.pickle_path = pickle_output_path
        self.num_padding_frames = padding_length
        # pick 1 every `sample_period` frames
        self.sampling_period = sampling_period

    def extract_2d_pose_from_annotation_file(self, annotation_path: str):
        self.annotation_path = Path(annotation_path)
        self.annotation_df = pd.read_csv(str(self.annotation_path))
        self.annotation_df = self.annotation_df.set_index(["participant_id", "file_id"])

        frame_info_pickle_path = Path(self.pickle_path) / "frame_info.pkl"
        annotation_pickle_path = Path(self.pickle_path) / "annotation.pkl"
        pose2D_pickle_path = Path(self.pickle_path) / "pose2D.pkl"

        frame_info = []
        pose_2d_results = []
        current_video_name = None
        num_rows = self.annotation_df.shape[0]
        annotation_count = 0
        frame_index = 0
        start_time = datetime.datetime.now()

        for index, row in self.annotation_df.iterrows():
            if (annotation_count + 1) % 50 == 0:
                end_time = datetime.datetime.now()
                total_seconds = (end_time - start_time).total_seconds()
                print(
                    f"{((annotation_count + 1) / num_rows * 100):.2f}% | "
                    f"Time used = {total_seconds//3600:02.0f}:{(total_seconds%3600) // 60:02.0f}:{total_seconds%60:02.0f}"
                )
            annotation_count += 1
            participant_id, video_name = index
            if current_video_name != video_name:
                current_video_name = video_name
                video_file = str(
                    Path(self.video_root_path)
                    / self.annotation_path.parent.name
                    / f"{current_video_name}.mp4"
                )
                cap = cv2.VideoCapture(video_file)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            images = []
            start = row["frame_start"] - (self.num_padding_frames * self.sampling_period)
            end = row["frame_end"] + (self.num_padding_frames * self.sampling_period)
            annotated_frame_start = row["frame_start"]
            annotated_frame_end = row["frame_end"]
            if start < 0:
                # pad further at the end
                end = end + abs(start)
                start = 0
            elif end > total_frames:
                # pad further at the start
                start = start - (end - total_frames)
                end = total_frames
            for frame_number in range(start, end, self.sampling_period):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                _, image = cap.read()
                images.append(image)
            bboxes, pose_2d_list, keypoint_2d_scores = self.pose_estimator_2d.inference(
                images
            )
            current_frame_info = []
            current_pose_2d_results = []
            for idx, image in enumerate(images):
                current_frame_info.append(
                    dict(
                        frame_index=frame_index,
                        participant_id=participant_id,
                        video_name=video_name,
                        annotation_id=row["annotation_id"],
                        frame=start + idx,
                        activity=row["activity"],
                    )
                )
                current_pose_2d_results.append(
                    dict(
                        frame_index=frame_index,
                        pose_2d=pose_2d_list[idx],
                        pose_2d_score=keypoint_2d_scores[idx],
                        pose_2d_avg_score=np.mean(keypoint_2d_scores[idx]),
                    )
                )
                frame_index += 1
            for item in current_frame_info:
                item['annotated_frame_start'] = annotated_frame_start
                item['annotated_frame_end'] = annotated_frame_end
            frame_info += current_frame_info
            pose_2d_results += current_pose_2d_results
            with open(frame_info_pickle_path, "wb") as f:
                pickle.dump(frame_info, f)
            with open(pose2D_pickle_path, "wb") as f:
                pickle.dump(pose_2d_results, f)

        annotation_df = pd.DataFrame.from_records(frame_info)\
            .groupby(['participant_id', 'video_name', 'annotation_id'])\
            .agg({
                'activity': 'first',
                'annotated_frame_start': 'first',
                'annotated_frame_end': 'first',
                'frame_index': ['min', 'max']
            }).reset_index()
        annotation_df.columns = annotation_df.columns.map(lambda x: '_'.join([str(i) for i in x]))
        annotation_df = annotation_df.rename(columns={
            'participant_id_': 'participant_id',
            'video_name_': 'video_name',
            'annotation_id_': 'annotation_id',
            'activity_first': 'activity',
            'annotated_frame_start_first': 'annotated_frame_start',
            'annotated_frame_end_first': 'annotated_frame_end',
            'frame_index_min': 'frame_index_start',
            'frame_index_max': 'frame_index_end'
        })
        annotation_df['frame_index_end'] = annotation_df['frame_index_end'] + 1
        with open(annotation_pickle_path, 'wb') as f:
            pickle.dump(annotation_df, f)
