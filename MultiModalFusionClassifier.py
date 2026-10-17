"""Multimodal 3D ViT response classifier with treatment-conditioned FiLM."""
import torch
from torch import nn
from monai.networks.nets import ViT

class MultiModalFusionClassifier(nn.Module):

    def __init__(self, *, modalities, num_classes, extra_dim, cuda_device, cuda_device2, vit_img_size, vit_patch_size, vit_hidden_size, vit_mlp_dim, vit_num_layers, vit_num_heads, vit_dropout_rate, token_layers, token_num_heads, token_ffn_dim, token_dropout, classifier_dropout_1, classifier_dropout_2):
        super().__init__()
        self.modalities = modalities
        self.hidden_size = vit_hidden_size
        self.primary_device = cuda_device
        self.device_map = {mod: cuda_device if i % 2 == 0 else cuda_device2 for i, mod in enumerate(modalities)}
        self.loco_film = nn.Sequential(nn.Linear(extra_dim, 128), nn.ReLU(), nn.Linear(128, self.hidden_size * 2)).to(self.primary_device)
        self.encoders = nn.ModuleDict()
        for i, mod in enumerate(modalities):
            encoder = ViT(in_channels=1, img_size=vit_img_size, patch_size=vit_patch_size, hidden_size=self.hidden_size, mlp_dim=vit_mlp_dim, num_layers=vit_num_layers, num_heads=vit_num_heads, dropout_rate=vit_dropout_rate, spatial_dims=3, classification=False).to(self.device_map[mod])
            self.encoders[mod] = encoder
        self.global_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size, device=self.primary_device))
        nn.init.trunc_normal_(self.global_token, std=0.02)
        token_encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=token_num_heads, dim_feedforward=token_ffn_dim, dropout=token_dropout, activation='gelu', batch_first=True).to(self.primary_device)
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer=token_encoder_layer, num_layers=token_layers).to(self.primary_device)
        self.classifier = nn.Sequential(nn.Dropout(p=classifier_dropout_1), nn.Linear(self.hidden_size, 128), nn.ReLU(), nn.Dropout(p=classifier_dropout_2), nn.Linear(128, num_classes)).to(self.primary_device)

    def forward(self, inputs, tabular):
        all_features = []
        for i, mod in enumerate(self.modalities):
            x = inputs[mod].to(self.device_map[mod])
            features = self.encoders[mod](x)
            if isinstance(features, tuple):
                features = features[0]
            all_features.append(features.to(self.primary_device))
        fused_feat = torch.cat(all_features, dim=1)
        batch_size = fused_feat.size(0)
        global_token = self.global_token.expand(batch_size, -1, -1)
        mri_only_feat = torch.cat([global_token, fused_feat], dim=1)
        mri_only_feat = self.fusion_transformer(mri_only_feat)
        shared_features = mri_only_feat[:, 0, :]
        gamma, beta = torch.chunk(self.loco_film(tabular), 2, dim=-1)
        return self.classifier(gamma * shared_features + beta)
