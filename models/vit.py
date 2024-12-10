import torch
import torch.nn as nn
from timm.models.helpers import build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model

__all__ = [
    "ViT3D",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (1, 22, 64, 64),
        "pool_size": None,
        "crop_pct": 0.875,
        "interpolation": "trilinear",
        "mean": (0.5,),
        "std": (0.5,),
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "vit3d_base_patch8": _cfg(
        url="path_to_your_pretrained_weights/vit3d_base_patch8.pth"
    ),
}


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for ViT3D"""

    def __init__(self, img_size=(22, 64, 64), patch_size=8, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # Calculate the number of patches in each dimension
        self.grid_size = (
            img_size[0] // patch_size, img_size[1] // patch_size, img_size[2] // patch_size)
        self.num_patches = self.grid_size[0] * \
            self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Apply Conv3D
        # Flatten spatial dimensions and permute
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT3D(VisionTransformer):
    def __init__(
        self,
        img_size=(22, 96, 96),  # Update to match actual input
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=19,
        in_chans=1,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        **kwargs,
    ):
        super(ViT3D, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            **kwargs,
        )

        # Replace patch_embed with PatchEmbed3D
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Correct pos_embed to match the number of patches generated by PatchEmbed3D
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + 1, embed_dim))  # Adjust for cls_token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x):
        x = self.patch_embed(x)  # Apply the 3D patch embedding
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # Extract only the cls_token for classification

    def forward(self, x):
        # Expected output: (batch_size, embed_dim)
        x = self.forward_features(x)
        # print(f"Shape before classification head: {x.shape}")  # Should be (batch_size, embed_dim)
        # Expected to match (batch_size, num_classes), e.g., (32, 19)
        x = self.head(x)
        return x


@register_model
def vit3d_base_patch8(pretrained=True, **kwargs):
    """ViT-Base model with 8x8x8 patch size and image size (22, 96, 96)."""
    model = ViT3D(
        img_size=(22, 96, 96),  # Update to match actual input
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        in_chans=1,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        **kwargs,
    )
    return model


@register_model
def vit3d_large_patch8(pretrained=False, **kwargs):
    """ViT-Base model with 8x8x8 patch size and image size (22, 96, 96)."""
    model = ViT3D(
        img_size=(22, 96, 96),
        patch_size=8,
        embed_dim=1024,  # Larger embed_dim
        depth=32,  # Increased depth
        num_heads=16,  # More attention heads
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=19,
        in_chans=1,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        **kwargs,
    )
    return model
