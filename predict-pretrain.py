"""
U-BDD++ Evaluation with pre-trained CLIP
"""
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
import torchvision.transforms as T
import numpy as np
import cv2
from tqdm import tqdm

from segment_anything import SamPredictor, sam_model_registry
import clip

from datasets.xbddataset import XBDDataset
from models.clipmlp.clipmlp import clip_prediction_ensemble, CONTRASTIVE_PROMPTS
from models.dino.util.slconfig import SLConfig
from models.dino.models.registry import MODULE_BUILD_FUNCS
from models.dino.util import box_ops
from utils.filters import preliminary_filter
from utils.utils import pixel_f1_iou

# Constants
IMAGE_WIDTH = 1024
DINO_TEXT_PROMPT = "building"
DAMAGE_DICT_BGR = [[0, 0, 0], [70, 172, 0], [0, 140, 253]]


def build_dino_model(args):
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)  # type: ignore
    return model, criterion, postprocessors


def load_dino_model(model_config_path, model_checkpoint_path, device="cpu"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model, criterion, postprocessors = build_dino_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, criterion, postprocessors


def get_dino_output(model, image, dino_threshold, postprocessors, device):
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None])

    outputs = postprocessors["bbox"](outputs, torch.Tensor([[1.0, 1.0]]).to(device))[0]
    scores = outputs["scores"]
    boxes = box_ops.box_xyxy_to_cxcywh(outputs["boxes"])
    select_mask = scores > dino_threshold

    pred_dict = {
        "boxes": boxes[select_mask],
        "scores": scores[select_mask],
        "labels": [DINO_TEXT_PROMPT] * len(scores[select_mask]),
    }
    return pred_dict


# Setting: Fine-tuned DINO + SAM + Pre-trained CLIP
def ubdd_plusplus(
    dino_model,
    dino_postprocessors,
    sam_predictor,
    clip_text,
    clip_model,
    clip_preprocess,
    clip_min_patch_size,
    clip_img_padding,
    dino_threshold,
    save_annotations,
    dataloader,
    device,x``
):
    f1_total = torch.zeros(3).to(device)
    iou_total = torch.zeros(3).to(device)
    count = 0
    with tqdm(dataloader, total=len(dataloader)) as pbar:
        for batch in pbar:
            output = get_dino_output(
                dino_model,
                batch["pre_image"][0],
                dino_threshold,
                dino_postprocessors,
                device=device,
            )
            boxes = output["boxes"].detach().cpu()  # cxcywh
            logits = output["scores"].detach().cpu()
            phrases = output["labels"]

            # boxes, logits, phrases = preliminary_filter(
            #     boxes, logits, phrases, dim_threshold=0.8, area_threshold=0.8
            # )
            boxes = box_convert(boxes * IMAGE_WIDTH, "cxcywh", "xyxy")

            # SAM prediction for all bounding boxes
            source_image = (
                (batch["pre_image_original"][0].permute(1, 2, 0) * 255)
                .numpy()
                .astype(np.uint8)
            )
            sam_predictor.set_image(source_image)

            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                boxes, source_image.shape[:2]
            ).to(device)

            if len(transformed_boxes) == 0:
                # no bounding box predictions
                masks = torch.zeros(
                    (1, 1, source_image.shape[0], source_image.shape[1]),
                    dtype=torch.uint8,
                ).to(device)
            else:
                with torch.no_grad():
                    masks, _, _ = sam_predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )

            # CLIP Prediction for each bounding box
            predictions = []
            for bbox in boxes.tolist():
                # Crop out the image given bboxes
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1

                # Apply the image buffer
                image_buffer_x = (
                    (clip_min_patch_size - w) / 2.0
                    if w < clip_min_patch_size
                    else clip_img_padding
                )
                image_buffer_y = (
                    (clip_min_patch_size - h) / 2.0
                    if h < clip_min_patch_size
                    else clip_img_padding
                )

                # Add padding for prediction
                x1_pad = max(int(x1 - image_buffer_x), 0)
                y1_pad = max(int(y1 - image_buffer_y), 0)
                x2_pad = min(int(x2 + image_buffer_x), IMAGE_WIDTH)
                y2_pad = min(int(y2 + image_buffer_y), IMAGE_WIDTH)

                pre_building_patch = T.ToPILImage()(
                    batch["pre_image_original"][0, :, y1_pad:y2_pad, x1_pad:x2_pad]
                )
                pre_building_patch_clip = (
                    clip_preprocess(pre_building_patch).unsqueeze(0).to(device)
                )
                post_building_patch = T.ToPILImage()(
                    batch["post_image_original"][0, :, y1_pad:y2_pad, x1_pad:x2_pad]
                )
                post_building_patch_clip = (
                    clip_preprocess(post_building_patch).unsqueeze(0).to(device)
                )
                pred = clip_prediction_ensemble(
                    clip_model,
                    pre_building_patch_clip,
                    post_building_patch_clip,
                    clip_text,
                )
                predictions.append(pred + 1)

            if len(predictions) == 0:
                pred_mask = masks[0].squeeze(0)
            else:
                # 0: background, 1: undamaged, 2: damaged
                pred_mask = (
                    (
                        masks.mul(
                            torch.tensor(predictions).to(device).reshape(-1, 1, 1, 1)
                        )
                    )
                    .max(dim=0)[0]
                    .squeeze(0)
                )

            file_name = batch["pre_file_name"][0].split("/")[-1][:-4]
            if save_annotations:
                pred_mask_annotate = pred_mask.cpu().numpy()
                color_mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
                for i in range(3):
                    color_mask[pred_mask_annotate == i] = DAMAGE_DICT_BGR[i]
                cv2.imwrite(f"outputs/test/{file_name}_ftdino_color.png", color_mask)
                exit(0)

            # 0: background, 1: undamaged, 2: damaged 3: unclassified
            gt_mask = (batch["post_image_mask"][0] * 255).type(torch.uint8).to(device)
            gt_mask = torch.where(
                (gt_mask > 1) & (gt_mask < 5), 2, torch.where(gt_mask >= 5, 3, gt_mask)
            )
            f1_scores, iou_scores = pixel_f1_iou(pred_mask, gt_mask, num_classes=3)
            f1_total += f1_scores
            iou_total += iou_scores
            count += 1

            pbar.set_postfix(
                # f1=f"{f1_scores}",
                mean_f1=f"{f1_total/count}",
                mf1=f"{(f1_total/count).mean()}",
                # mean_iou=f"{iou_total/count}",
                miou=f"{(iou_total/count).mean()}",
                refresh=False,
            )

        print(f"Mean F1: {f1_total/count}")
        print(f"mF1: {(f1_total/count).mean()}")
        print(f"Mean IoU: {iou_total/count}")
        print(f"mIoU: {(iou_total/count).mean()}")


def get_args():
    parser = argparse.ArgumentParser(description="Prediction of U-BDD++ on xBD dataset")
    parser.add_argument(
        "--test-set-path",
        "-tssp",
        type=str,
        required=True,
        help="Path to the test set directory",
        dest="test_set_path",
    )
    parser.add_argument(
        "--clip-min-patch-size",
        "-cmps",
        type=int,
        default=100,
        help="Minimum patch size for CLIP",
        dest="clip_min_patch_size",
    )
    parser.add_argument(
        "--clip-img-padding",
        "-cip",
        type=int,
        default=10,
        help="Padding of patch for CLIP",
        dest="clip_img_padding",
    )
    parser.add_argument(
        "--dino-path",
        "-dp",
        type=str,
        required=True,
        help="Path to the DINO model",
        dest="dino_path",
    )
    parser.add_argument(
        "--dino-config",
        "-dc",
        type=str,
        required=True,
        help="Path to the DINO config file",
        dest="dino_config",
    )
    parser.add_argument(
        "--dino-threshold",
        "-dt",
        type=float,
        default=0.15,
        help="Threshold for DINO bounding box prediction",
        dest="dino_threshold",
    )
    parser.add_argument(
        "--sam-path",
        "-sp",
        type=str,
        required=True,
        help="Path to the SAM model",
        dest="sam_path",
    )
    parser.add_argument(
        "--save-annotations",
        "-sa",
        action="store_true",
        help="Save annotations",
        dest="save_annotations",
    )
    return parser.parse_args()


def main():
    args = get_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    dino_model, criterion, postprocessors = load_dino_model(
        args.dino_config, args.dino_path, device=device_str
    )

    xbd_dataset = XBDDataset(
        [args.test_set_path], dino_transform=True, include_masks=True, include_post=True
    )
    xbd_dataloader = DataLoader(xbd_dataset, batch_size=1, shuffle=False, num_workers=4)
    sam_model = sam_model_registry["default"](checkpoint=args.sam_path).to(device)
    sam_predictor = SamPredictor(sam_model)
    clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device_str)
    clip_text = clip.tokenize(CONTRASTIVE_PROMPTS).to(device_str)

    ubdd_plusplus(
        dino_model,
        postprocessors,
        sam_predictor,
        clip_text,
        clip_model,
        clip_preprocess,
        args.clip_min_patch_size,
        args.clip_img_padding,
        args.dino_threshold,
        args.save_annotations,
        xbd_dataloader,
        device_str,
    )


if __name__ == "__main__":
    main()
