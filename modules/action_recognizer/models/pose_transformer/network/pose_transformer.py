import torch
import torch.functional as F
import torch.nn as nn


# Modified from modules/action_recognizer/models/action_transformer/action_transformer.py
# to use each keypoints of all frames as tokens.
class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0, is_pre_norm=True):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.is_pre_norm = is_pre_norm
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.is_pre_norm:
            inp_x = self.layer_norm_1(x)
            x = x + self.attn(inp_x, inp_x, inp_x)[0]
            x = x + self.linear(self.layer_norm_2(x))
        else:
            x = x + self.attn(x, x, x)[0]
            x = self.layer_norm_1(x)
            x = x + self.linear(x)
            x = self.layer_norm_2(x)
        return x
    

class PoseTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, num_joints, num_frames, dropout=0.0, is_pre_norm=True):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (2 for 2D poses, 3 for 3D poses)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            num_joints - Number of joints in a pose
            num_frames - Number of (maximum) frame per action
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.num_joints = num_joints

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels, embed_dim)
        self.transformer = nn.Sequential(
            *[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout, is_pre_norm=is_pre_norm) for _ in range(num_layers)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_frames * num_joints, embed_dim))


    def forward(self, x):
        # Preprocess input
        B, T, _ = x.shape
        x = self.input_layer(x)
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
