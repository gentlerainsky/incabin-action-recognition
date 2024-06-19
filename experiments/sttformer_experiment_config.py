# Model
from modules.action_recognizer.models.action_transformer.lit_action_transformer import LitActionTransformer
from modules.action_recognizer.models.pose_transformer.lit_pose_transformer import LitPoseTransformer
from modules.action_recognizer.models.stgcn.lit_stgcn import LitSTGCN
from modules.action_recognizer.models.STTFormer.lit_sttformer import LitSTTFormer


from modules.action_recognizer.dataset.pose_dataset import all_activities

sttformer_2D = dict(
    name = 'sttformer_2d',
    saved_path = 'sttformer',
    lit_model = LitSTTFormer(
        dict(
            len_parts=6,
            num_classes=len(all_activities),
            num_joints=13,
            num_frames=30,
            num_heads=3,
            num_persons=1,
            kernel_size=[3, 5],
            config=[
                [64,  64,  16], [64,  64,  16], 
                [64,  128, 32], [128, 128, 32],
                [128, 256, 64], [256, 256, 64], 
                [256, 256, 64], [256, 256, 64]
            ]
        ),                  
        lr=1e-3,
        is_pose_3d=False
    )
)

sttformer_3D = dict(
    name = 'sttformer_3d',
    saved_path = 'sttformer',
    lit_model = LitSTTFormer(
        dict(
            len_parts=6,
            num_classes=len(all_activities),
            num_joints=13,
            num_frames=30,
            num_heads=3,
            num_persons=1,
            kernel_size=[3, 5],
            config=[
                [64,  64,  16], [64,  64,  16], 
                [64,  128, 32], [128, 128, 32],
                [128, 256, 64], [256, 256, 64], 
                [256, 256, 64], [256, 256, 64]
            ]
        ),                  
        lr=1e-3,
        is_pose_3d=True
    )
)

sttformer_3D_bone = dict(
    name = 'sttformer_3d_bone',
    saved_path = 'sttformer',
    batch_size=128,
    lit_model = LitSTTFormer(
        dict(
            len_parts=6,
            num_classes=len(all_activities),
            num_joints=13 + 15,
            num_frames=30,
            num_heads=3,
            num_persons=1,
            kernel_size=[3, 5],
            config=[
                [64,  64,  16], [64,  64,  16], 
                [64,  128, 32], [128, 128, 32],
                [128, 256, 64], [256, 256, 64], 
                [256, 256, 64], [256, 256, 64]
            ]
        ),                  
        lr=1e-3,
        is_pose_3d=True,
        use_bone=True
    )
)
sttformer_3D_velo = dict(
    name = 'sttformer_3d_velo',
    saved_path = 'sttformer',
    batch_size=128,
    lit_model = LitSTTFormer(
        dict(
            len_parts=6,
            num_classes=len(all_activities),
            num_joints=13 + 13,
            num_frames=30,
            num_heads=3,
            num_persons=1,
            kernel_size=[3, 5],
            config=[
                [64,  64,  16], [64,  64,  16], 
                [64,  128, 32], [128, 128, 32],
                [128, 256, 64], [256, 256, 64], 
                [256, 256, 64], [256, 256, 64]
            ]
        ),                  
        lr=1e-3,
        is_pose_3d=True,
        use_velocity=True
    )
)

sttformer_3D_bone_velo = dict(
    name = 'sttformer_3d_bone_velo',
    saved_path = 'sttformer',
    batch_size=64,
    lit_model = LitSTTFormer(
        dict(
            len_parts=6,
            num_classes=len(all_activities),
            num_joints=13 + 13 + 15,
            num_frames=30,
            num_heads=3,
            num_persons=1,
            kernel_size=[3, 5],
            config=[
                [64,  64,  16], [64,  64,  16], 
                [64,  128, 32], [128, 128, 32],
                [128, 256, 64], [256, 256, 64], 
                [256, 256, 64], [256, 256, 64]
            ]
        ),                  
        lr=1e-3,
        is_pose_3d=True,
        use_bone=True,
        use_velocity=True,
    )
)

experiment_config = [
    sttformer_2D,
    sttformer_3D,
    sttformer_3D_bone,
    sttformer_3D_velo,
    sttformer_3D_bone_velo
]
