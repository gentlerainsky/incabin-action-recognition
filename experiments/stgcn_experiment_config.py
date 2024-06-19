# Model
from modules.action_recognizer.models.action_transformer.lit_action_transformer import LitActionTransformer
from modules.action_recognizer.models.pose_transformer.lit_pose_transformer import LitPoseTransformer
from modules.action_recognizer.models.stgcn.lit_stgcn import LitSTGCN
from modules.action_recognizer.models.STTFormer.lit_sttformer import LitSTTFormer


from modules.action_recognizer.dataset.pose_dataset import all_activities

stgcn_2D = dict(
    name = 'stgcn_2d',
    saved_path = 'stgcn',
    lit_model = LitSTGCN(
        dict(
            in_channels=2,
            num_class=len(all_activities),
            graph_args=dict(
                layout='drive_and_act'
            ),
            edge_importance_weighting=False,
        ),
        lr=1e-3,
        is_pose_3d=False,
        num_frames=30
    )
)

stgcn_3D = dict(
    name = 'stgcn_2d',
    saved_path = 'stgcn',
    lit_model = LitSTGCN(
        dict(
            in_channels=2,
            num_class=len(all_activities),
            graph_args=dict(
                layout='drive_and_act'
            ),
            edge_importance_weighting=False,
        ),
        lr=1e-3,
        is_pose_3d=True,
        num_frames=30
    )
)

experiment_config = [
    stgcn_2D,
    stgcn_3D
]
