# Model
from modules.action_recognizer.models.action_transformer.lit_action_transformer import LitActionTransformer
from modules.action_recognizer.models.pose_transformer.lit_pose_transformer import LitPoseTransformer
from modules.action_recognizer.models.stgcn.lit_stgcn import LitSTGCN
from modules.action_recognizer.models.STTFormer.lit_sttformer import LitSTTFormer


from modules.action_recognizer.dataset.pose_dataset import all_activities

action_transformer_2D_M = dict(
    name = 'action_transformer_2d_M',
    saved_path = 'action_transformer',
    lit_model = LitActionTransformer(
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
    )
)

action_transformer_2D_L = dict(
    name = 'action_transformer_2d_L',
    saved_path = 'action_transformer',
    lit_model = LitActionTransformer(
        dict(
            embed_dim=256,
            hidden_dim=512,
            num_heads=4,
            num_layers=6,
            num_classes=len(all_activities),
            num_joints=13,
            num_frames=30,
            dropout=0.3,
            is_pre_norm=False
        ),
        lr=1e-3,
        is_pose_3d=False,
    )
)

action_transformer_3D_M = dict(
    name = 'action_transformer_3d_M',
    saved_path = 'action_transformer',
    lit_model = LitActionTransformer(
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

action_transformer_3D_L = dict(
    name = 'action_transformer_3d_L',
    saved_path = 'action_transformer',
    lit_model = LitActionTransformer(
        dict(
            embed_dim=256,
            hidden_dim=512,
            num_heads=4,
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


action_transformer_3D_M_bone = dict(
    name = 'action_transformer_3d_M_bone',
    saved_path = 'action_transformer',
    lit_model = LitActionTransformer(
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
        use_bone=True,
    )
)

action_transformer_3D_L_bone = dict(
    name = 'action_transformer_3d_L_bone',
    saved_path = 'action_transformer',
    lit_model = LitActionTransformer(
        dict(
            embed_dim=256,
            hidden_dim=512,
            num_heads=4,
            num_layers=6,
            num_classes=len(all_activities),
            num_joints=13 + 15,
            num_frames=30,
            dropout=0.3,
            is_pre_norm=False
        ),
        lr=1e-3,
        is_pose_3d=True,
        use_bone=True,
    )
)


action_transformer_3D_M_velo = dict(
    name = 'action_transformer_3d_M_velo',
    saved_path = 'action_transformer',
    lit_model = LitActionTransformer(
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
        use_velocity=True,
    )
)

action_transformer_3D_L_velo = dict(
    name = 'action_transformer_3d_L_velo',
    saved_path = 'action_transformer',
    lit_model = LitActionTransformer(
        dict(
            embed_dim=256,
            hidden_dim=512,
            num_heads=4,
            num_layers=6,
            num_classes=len(all_activities),
            num_joints=13 + 13,
            num_frames=30,
            dropout=0.3,
            is_pre_norm=False
        ),
        lr=1e-3,
        is_pose_3d=True,
        use_velocity=True,
    )
)

action_transformer_3D_M_bone_velo = dict(
    name = 'action_transformer_3d_M_bone_velo',
    saved_path = 'action_transformer',
    lit_model = LitActionTransformer(
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
        use_velocity=True,
    )
)

action_transformer_3D_L_bone_velo = dict(
    name = 'action_transformer_3d_L_bone_velo',
    saved_path = 'action_transformer',
    lit_model = LitActionTransformer(
        dict(
            embed_dim=256,
            hidden_dim=512,
            num_heads=4,
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
        use_velocity=True,
    )
)

experiment_config = [
    action_transformer_2D_M,
    action_transformer_2D_L,
    action_transformer_3D_M,
    action_transformer_3D_L,
    action_transformer_3D_M_bone,
    action_transformer_3D_L_bone,
    action_transformer_3D_M_velo,
    action_transformer_3D_L_velo,
    action_transformer_3D_M_bone_velo,
    action_transformer_3D_L_bone_velo
]
