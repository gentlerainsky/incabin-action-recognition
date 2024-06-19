# Model
from modules.action_recognizer.models.action_transformer.lit_action_transformer import LitActionTransformer
from modules.action_recognizer.models.pose_transformer.lit_pose_transformer import LitPoseTransformer
from modules.action_recognizer.models.stgcn.lit_stgcn import LitSTGCN
from modules.action_recognizer.models.STTFormer.lit_sttformer import LitSTTFormer


from modules.action_recognizer.dataset.pose_dataset import all_activities

pose_transformer_2D = dict(
    name = 'pose_transformer_2d',
    saved_path = 'pose_transformer',
    batch_size=128,
    lit_model = LitPoseTransformer(
        dict(
            embed_dim=192,
            hidden_dim=256,
            num_heads=3,
            num_layers=6,
            num_classes=len(all_activities),
            num_joints=13,
            num_frames=30,
            dropout=0.3,
            is_pre_norm=False
        ),
        lr=1e-3,
        is_pose_3d=False,
    ),                  
    lr=1e-3,
    is_pose_3d=False
)

pose_transformer_3D = dict(
    name = 'pose_transformer_3d',
    saved_path = 'pose_transformer',
    batch_size=128,
    lit_model = LitPoseTransformer(
        dict(
            embed_dim=192,
            hidden_dim=256,
            num_heads=3,
            num_layers=6,
            num_classes=len(all_activities),
            num_joints=13,
            num_frames=30,
            dropout=0.3,
            is_pre_norm=False
        ),
        lr=1e-3,
        is_pose_3d=True,
    )
)

pose_transformer_3D_bone = dict(
    name = 'pose_transformer_3d_bone',
    saved_path = 'pose_transformer',
    batch_size=64,
    lit_model = LitPoseTransformer(
        dict(
            embed_dim=192,
            hidden_dim=256,
            num_heads=3,
            num_layers=6,
            num_classes=len(all_activities),
            num_joints=13 + 15,
            num_frames=30,
            dropout=0.3,
            is_pre_norm=False
        ),
        lr=1e-3,
        is_pose_3d=True,
        use_bone=True
    )
)
pose_transformer_3D_velo = dict(
    name = 'pose_transformer_3d_velo',
    saved_path = 'pose_transformer',
    batch_size=64,
    lit_model = LitPoseTransformer(
        dict(
            embed_dim=192,
            hidden_dim=256,
            num_heads=3,
            num_layers=6,
            num_classes=len(all_activities),
            num_joints=13 + 13,
            num_frames=30,
            dropout=0.3,
            is_pre_norm=False
        ),
        lr=1e-3,
        is_pose_3d=True,
        use_velocity=True
    )
)

pose_transformer_3D_bone_velo = dict(
    name = 'pose_transformer_3d_bone_velo',
    saved_path = 'pose_transformer',
    batch_size=32, 
    lit_model = LitPoseTransformer(
        dict(
            embed_dim=192,
            hidden_dim=256,
            num_heads=3,
            num_layers=6,
            num_classes=len(all_activities),
            num_joints=13 + 15 + 13,
            num_frames=30,
            dropout=0.3,
            is_pre_norm=False
        ),
        lr=1e-3,
        is_pose_3d=True,
        use_bone=True,
        use_velocity=True
    )
)

experiment_config = [
    pose_transformer_2D,
    pose_transformer_3D,
    pose_transformer_3D_bone,
    pose_transformer_3D_velo,
    pose_transformer_3D_bone_velo
]
 