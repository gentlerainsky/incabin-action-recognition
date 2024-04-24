import numpy as np
from mmpose.apis import MMPoseInferencer


## DEFINE MODELS
### Faster RCNN for Human Detection
# mmdet_config_path = 'mmdet::faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'
# mmdet_config_path = 'faster-rcnn_r50-caffe_fpn_ms-1x_coco-person'
# mmdet_model_weight = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'\
# 'faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'

mmdet_config_path = "./modules/pose_extractor/pose2d_estimator/config/faster_rcnn.py"
mmdet_model_weight = (
    "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/"
    "faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
)

### HR-Net for 2D Pose Estimation
# mmpose_config_path = 'mmpose::body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py'
mmpose_config_path = "td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288"
mmpose_model_weight = (
    "https://download.openmmlab.com/mmpose/v1/"
    "body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth"
)


class Pose2DEstimator:
    def __init__(self):
        self.mmpose_inferencer = MMPoseInferencer(
            pose2d=mmpose_config_path,
            pose2d_weights=mmpose_model_weight,
            det_model=mmdet_config_path,
            det_weights=mmdet_model_weight,
            det_cat_ids=[0],
        )

    def inference(self, images):
        mmpose_results = self.mmpose_inferencer(images)
        bboxes = []
        pose_2d_list = []
        keypoint_2d_scores = []
        for idx, mmpose_result in enumerate(mmpose_results):
            image_results = mmpose_result["predictions"][0]
            best_box_idx = np.argmax([box["bbox_score"] for box in image_results])
            detected = image_results[best_box_idx]
            bbox = detected["bbox"]
            pose_2d = detected["keypoints"]
            bboxes.append(bbox)
            pose_2d_list.append(pose_2d)
            keypoint_2d_scores.append(detected["keypoint_scores"])
        return bboxes, pose_2d_list, keypoint_2d_scores
