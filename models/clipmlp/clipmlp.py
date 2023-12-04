import torch
from torch import nn
import numpy as np

POSITIVE_PROMPTS = [
    "A satellite photo of a building",
    "normal building",
    "undamaged building",
    "building",
]
NEGATIVE_PROMPTS = [
    "A satellite photo of a ruin",
    "damaged building",
    "destroyed building",
    "ruin",
]
CONTRASTIVE_PROMPTS = POSITIVE_PROMPTS + NEGATIVE_PROMPTS

# Prompt Ensemble
def clip_prediction_ensemble(clip_model, pre_patch, post_patch, texts):
    with torch.no_grad():
        pre_logits_per_image, pre_logits_per_text = clip_model(pre_patch, texts)
        post_logits_per_image, post_logits_per_text = clip_model(post_patch, texts)

        # find the largest logits
        max_values_pre = torch.tensor(
            [
                pre_logits_per_image[0, : len(POSITIVE_PROMPTS)].max(),
                pre_logits_per_image[0, -len(POSITIVE_PROMPTS) :].max(),
            ],
            dtype=torch.float32,
        )
        max_values_post = torch.tensor(
            [
                post_logits_per_image[0, : len(POSITIVE_PROMPTS)].max(),
                post_logits_per_image[0, -len(POSITIVE_PROMPTS) :].max(),
            ],
            dtype=torch.float32,
        )

        probs_pre = max_values_pre.softmax(dim=-1).cpu().numpy()
        probs_post = max_values_post.softmax(dim=-1).cpu().numpy()

        calc = 0.5 * (probs_post[0] + 0.01 - probs_pre[0]) + 0.5 * (
            probs_post[0] - probs_post[1]
        )
        pred = 0 if calc >= -0.4 else 1

    return pred


# post only
def clip_prediction_post(clip_model, _, post_patch, texts, true):
    with torch.no_grad():
        logits_per_image, logits_per_text = clip_model(post_patch, texts)

        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        pred = np.floor(np.argmax(probs) / (len(texts) / 2))
        # confidence = probs[0][pred]

    # return pred, confidence
    return pred, None


# positive prompt only
def clip_prediction_positive(clip_model, pre_patch, post_patch, texts, true):
    with torch.no_grad():
        pre_logits_per_image, pre_logits_per_text = clip_model(pre_patch, texts)
        post_logits_per_image, post_logits_per_text = clip_model(post_patch, texts)

        diff = pre_logits_per_image.max() - post_logits_per_image.max()
        # diff = pre_logits_per_image.sum() - post_logits_per_image.sum()
        # print(diff)
        pred = 0 if diff <= 2 else 1

    return pred, None


def clip_prediction_contrastive_old(clip_model, pre_patch, post_patch, texts, true):
    with torch.no_grad():
        pre_logits_per_image, pre_logits_per_text = clip_model(pre_patch, texts)
        post_logits_per_image, post_logits_per_text = clip_model(post_patch, texts)

        probs_pre = pre_logits_per_image.softmax(dim=-1).cpu().numpy()
        probs_post = post_logits_per_image.softmax(dim=-1).cpu().numpy()
        diff = probs_post - probs_pre
        pred = (
            0
            if diff[: len(POSITIVE_PROMPTS)].sum() > diff[len(POSITIVE_PROMPTS) :].sum()
            else 1
        )
    return pred, None


class CLIPMLP(nn.Module):
    def __init__(self, clip_model, output_dim):
        super().__init__()
        self.clip_model = clip_model
        self.fc_layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = self.clip_model.encode_image(x)
        x = x.to(torch.float32)
        x = self.fc_layers(x)
        return x
