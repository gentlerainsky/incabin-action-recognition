from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torchvision

from models.basemodel import Basemodel
from sam import SAM
from utils.misc import download_file_if_not_exist

import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# import apex

# https://download.pytorch.org/models/swin3d_s-da41c237.pth
# https://pytorch.org/vision/stable/_modules/torchvision/models/video/swin_transformer.html


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, use_flash_attn=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        # self.mask = mask
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.use_flash_attn = use_flash_attn  # use_flash_attn

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        #qkv = self.qkv_act(qkv)
        
        #assert not torch.isnan(qkv).any()
        

        if self.use_flash_attn:
            qkv = rearrange(qkv, "b n (c h d) -> b n c h d", h=self.heads, c=3)
            attn = flash_attn_qkvpacked_func(
                qkv, softmax_scale=self.scale, dropout_p=0, window_size=(-1, -1)
            )
            out = rearrange(attn, "b n h d -> b n (h d)")

        else:
            qkv = qkv.chunk(3, dim=-1)
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
            )
            q_k = torch.matmul(q, k.transpose(-1, -2))

            if mask is not None:
                q_k = q_k + mask  # torch.stack([mask] * self.heads, dim=1)
                # assert not torch.isnan(q_k).any()

            dots = q_k * self.scale

            attn = self.attend(dots)

            if mask is not None:
                attn = attn * mask

            attn = self.dropout(attn)

            out = torch.matmul(attn, v)

            out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, use_flash_attn=False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            use_flash_attn=use_flash_attn,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, mask=None):

        if torch.isnan(x).any():
            print("x is nan!")

        for i, (attn, ff) in enumerate(self.layers):
            #if i == 0:
            x = attn(x, mask) + x
            #else:
                #x = attn(x) + x
            #x = ff(x) + x
        return x


class PoseTransformer(Basemodel):
    def __init__(
        self,
        num_classes,
        sample_length,
        class_weights,
        classification_mode,
        N_KEYPOINTS,
        masking=False,
        scores=False,
        use_flash_attn=False,
        use_registers=0,
        **kwargs
    ):
        super(PoseTransformer, self).__init__(
            num_classes, sample_length, class_weights, classification_mode
        )

        # self.net = torchvision.models.video.swin3d_s(weights=None, num_classes=num_classes)

        # self.mlp = torchvision.ops.MLP(2 * 17 * 16, [1024, num_classes], dropout=0.2, activation_layer=torch.nn.ReLU)

        # input consists of 2 (coordinates) * 17 (keypoints) * 16 (frames)
        # input tensor is

        # k = 17
        #num_kp = 17 * 3
        self.use_n_registers = use_registers
        self.use_masking = masking
        self.use_cls_token = True
        self.pool = 'mean'
        if self.use_cls_token:
            self.pool = 'cls'

        #self.pool = 'cls'

        self.device = torch.device("cuda")

        # N_KEYPOINTS = 17  # 13
        N_CHANNELS = 2 + (1 if scores else 0)  # 3+1

        token_dim = N_CHANNELS * N_KEYPOINTS
        self.max_tokens = sample_length  # sample_length #* 3

        emb_dropout = 0.1
        dropout = 0.1

        token_embedding_dim = 128  # 128
        dim_head = 128
        mlp_dim = 1024

        # n_heads = [8] * 12
        self.n_heads = 8
        self.depth = 6

        self.emb_rearrange = Rearrange("b f v k c -> b (f v) (k c)")
        self.emb_linear = nn.Linear(token_dim, token_embedding_dim)
        self.emb_norm1 = nn.LayerNorm(token_dim)
        self.emb_norm2 = nn.LayerNorm(token_embedding_dim)

        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            # nn.Dropout(0.2),
            self.emb_rearrange,
            self.emb_norm1,
            self.emb_linear,
            self.emb_norm2,
        )

        if self.use_cls_token:
            self.max_tokens += 1
            self.cls_token = nn.Parameter(torch.randn(1, 1, token_embedding_dim))

        if self.use_n_registers > 0:
            self.registers = nn.Parameter(torch.randn(1, self.use_n_registers, token_embedding_dim))
            
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_tokens, token_embedding_dim))

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.max_tokens, token_embedding_dim)
        )

        self.dropout = nn.Dropout(emb_dropout)

        # mask = torch.triu(torch.full((tokens+1, tokens+1), float('-inf'), device=self.device), diagonal=1)
        # print(mask)
        # exit()

        self.transformer = Transformer(
            token_embedding_dim,
            self.depth,
            self.n_heads,
            dim_head,
            mlp_dim,
            dropout,
            use_flash_attn,
        )

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(token_embedding_dim),
            nn.Linear(token_embedding_dim, num_classes),
        )

        # self.fc = nn.Linear(400, num_classes)

    def training_step(self, train_batch):
        x, y, additional_data = train_batch

        valid_entries = None  # do not use valid entries
        if "valid_entries" in additional_data:
            valid_entries = additional_data["valid_entries"]

        outputs = self(x, valid_entries)
        # outputs = torch.functional.F.softmax(outputs, dim=-1)
        # print(torch.min(outputs), torch.max(outputs), outputs.shape)

        if self.classification_mode == "binary":
            # outputs = torch.squeeze(outputs) # reduce (batch, 1) -> (batch) for binary classification
            y = y.float()

        loss = self.lossfn(outputs, y)
        return outputs, loss

    def forward(self, inp, valid_samples=None):
        src = torch.stack(
            [
                inp[x][y]
                for x in inp.keys()
                if "pose" in x
                for y in inp[x].keys()
                if "keypoints" in y
            ],
            dim=2,
        )

        # src = repeat(src, 'b f k c -> b (f r) k c', r = 2)
        # src = src[:, 1:-1, :, :]

        use_masking = self.use_masking and valid_samples is not None
        # use_masking = True
        # mask=None

        # print(self.use_masking)

        if use_masking:
            comp_mask = torch.full(
                (src.shape[0], self.n_heads, self.max_tokens, self.max_tokens), 
                0, 
                device=self.device, dtype=torch.float32)
            comp_mask.requires_grad = False
            
            for batch_idx in range(len(valid_samples)):
                invalid_mask = ~valid_samples[batch_idx]

                if self.use_cls_token:
                    invalid_mask = torch.cat((torch.tensor([False]), invalid_mask), dim=0)

                comp_mask[batch_idx, :, :, invalid_mask] = float('-inf')
                comp_mask[batch_idx, :, invalid_mask, :] = float('-inf')

                # if self.use_cls_token:
                #     index_tensor = index_tensor + 1

                # cls_offset = 1 if self.use_cls_token else 0
                # comp_mask[batch_idx, :, :, index_tensor] = 0
                # comp_mask[batch_idx, :, index_tensor, :] = 0

            # if self.use_cls_token:
            #     comp_mask[:, :, 0, :] = 0
            #     comp_mask[:, :, :, 0] = 0
        else:
            comp_mask = None

        #     mask = torch.full((src.shape[0], self.max_tokens, self.max_tokens), float(0), device=self.device)

        #     if self.use_cls_token:
        #         valid_samples = nn.functional.pad(valid_samples, (1, 0, 0, 0), mode='constant', value=True)

        #     for i, mask_for_sample in enumerate(valid_samples):
        #         print(mask_for_sample)

        #         mask[i, ~mask_for_sample, :] = float('-inf')
        #         mask[i, :, ~mask_for_sample] = float('-inf')

        #         if self.use_cls_token:
        #             mask[i, 0, :] = 0 # TODO: analyze why this is needed
        #             mask[i, :, 0] = 0

        # print(mask[i, :])

        # mask = None

        # assert not torch.isnan(src).any()

        # 16 16 17 2
        x = self.to_patch_embedding(src)
        # x = self.emb_rearrange(src)
        # assert not torch.isnan(x).any()
        # x = self.emb_norm1(x)
        # assert not torch.isnan(x).any()
        # x = self.emb_linear(x)
        # assert not torch.isnan(x).any()
        # x = self.emb_norm2(x)
        # assert not torch.isnan(x).any()

        # assert not torch.isnan(x).any()

        b, n, _ = x.shape

        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, : (n + 1)]
        else:
            x += self.pos_embedding

        if self.use_n_registers > 0:
            registers_repeat = repeat(self.registers, '1 r d -> b r d', b = b)
            x = torch.cat((registers_repeat, x), dim=1)

        x = self.dropout(x)

        assert not torch.isnan(x).any()

        x = self.transformer(x, comp_mask)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #x = x[:, 0:self.use_n_registers].mean(dim = 1)

        x = self.to_latent(x)
        return self.mlp_head(x)

    # @override
    def configure_optimizers(self, lr):

        # lr = 1.0e-5 / math.sqrt(8.0)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        return optimizer

    def get_default_weights(self):
        return None

    def load_weights(self, weights):
        if weights is None:
            return
        loaded_state_dict = torch.load(weights)

        for key in list(
            loaded_state_dict.keys()
        ):  # ['net.head.weight', 'net.head.bias']:
            if "mlp_head" in key:
                loaded_state_dict.pop(key)

        self.load_state_dict(loaded_state_dict, strict=False)

    @staticmethod
    def get_required_features():
        return ["pose"]
