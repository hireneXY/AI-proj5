#!/usr/bin/env python3
import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models

class MISAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # encoders
        self.text_encoder = AutoModel.from_pretrained(getattr(config, "text_model_name", "bert-base-uncased"))
        text_feat_dim = getattr(self.text_encoder.config, "hidden_size", getattr(config, "text_feature_dim", 768))
        backbone = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        image_feat_dim = getattr(config, "image_feature_dim", 2048)
        # dims
        shared_dim = getattr(config, "misa_shared_dim", getattr(config, "shared_dim", 128))
        private_dim = getattr(config, "misa_private_dim", getattr(config, "private_dim", 128))
        # projections
        self.text_private = nn.Linear(text_feat_dim, private_dim)
        self.text_shared = nn.Linear(text_feat_dim, shared_dim)
        self.image_private = nn.Linear(image_feat_dim, private_dim)
        self.image_shared = nn.Linear(image_feat_dim, shared_dim)
        # classifier
        clf_in = shared_dim + private_dim * 2
        self.classifier = nn.Sequential(
            nn.Dropout(getattr(config, "dropout_rate", 0.3)),
            nn.Linear(clf_in, max(clf_in // 2, 64)),
            nn.ReLU(),
            nn.Dropout(getattr(config, "dropout_rate", 0.3)),
            nn.Linear(max(clf_in // 2, 64), getattr(config, "num_classes", 3))
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, image):
        batch_size = input_ids.size(0)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]
        t_priv = self.text_private(text_feat)
        t_shared = self.text_shared(text_feat)
        img_feat = self.image_encoder(image)
        img_feat = img_feat.view(batch_size, -1)
        i_priv = self.image_private(img_feat)
        i_shared = self.image_shared(img_feat)
        shared = (t_shared + i_shared) / 2.0
        combined = torch.cat([shared, t_priv, i_priv], dim=1)
        logits = self.classifier(combined)
        return logits, {
            "text_feat": text_feat,
            "image_feat": img_feat,
            "t_private": t_priv,
            "i_private": i_priv,
            "t_shared": t_shared,
            "i_shared": i_shared,
            "shared": shared,
            "combined": combined
        }

    def compute_misa_losses(self, features):
        mse = nn.MSELoss()
        t_feat = features["text_feat"]
        i_feat = features["image_feat"]
        t_priv = features["t_private"]
        i_priv = features["i_private"]
        t_shared = features["t_shared"]
        i_shared = features["i_shared"]
        if not hasattr(self, "text_reconstructor"):
            self.text_reconstructor = nn.Linear(t_priv.size(1) + t_shared.size(1), t_feat.size(1)).to(t_feat.device)
        if not hasattr(self, "image_reconstructor"):
            self.image_reconstructor = nn.Linear(i_priv.size(1) + i_shared.size(1), i_feat.size(1)).to(i_feat.device)
        rec_t = self.text_reconstructor(torch.cat([t_priv, t_shared], dim=1))
        rec_i = self.image_reconstructor(torch.cat([i_priv, i_shared], dim=1))
        recon_loss = 0.5 * (mse(rec_t, t_feat) + mse(rec_i, i_feat))
        sim_loss = mse(t_shared, i_shared)
        t_dot = (t_priv * t_shared).sum(dim=1)
        i_dot = (i_priv * i_shared).sum(dim=1)
        diff_loss = 0.5 * (t_dot.pow(2).mean() + i_dot.pow(2).mean())
        return {"reconstruction": recon_loss, "similarity": sim_loss, "difference": diff_loss}