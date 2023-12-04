from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import torch


def get_ap_score(detections, gt_annotations, total_images):
    # Create COCO object for groundtruth
    gt_coco = COCO()
    gt_coco.dataset = gt_annotations
    gt_coco.createIndex()

    # Create COCO object for detections
    dt_coco = gt_coco.loadRes(detections)

    # Create COCOEval object
    coco_eval = COCOeval(gt_coco, dt_coco, "bbox")

    # Select the image to evaluate
    # coco_eval.params.imgIds = [1]  # Replace 1 with your actual image id
    coco_eval.params.imgIds = list(
        range(total_images)
    )  # Replace 1 with your actual image id

    # Run COCO evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def pixel_f1_iou(pred: torch.Tensor, gt: torch.Tensor, num_classes=1):
    """Calculate pixel-wise f1 score. mask should be either:
    0: background, 1: building (positive), or
    0: background, 1: no-damage, 2: damaged, or
    0: background, 1: no-damage, 2: minor-damage, 3: major-damage, 4: destroyed
    """
    if num_classes == 1:
        tp = ((pred == 1) & (gt == 1)).sum()
        fp = ((pred == 1) & (gt == 0)).sum()
        fn = ((pred == 0) & (gt == 1)).sum()
        tn = ((pred == 0) & (gt == 0)).sum()

        # Handles empty predictions / ground truth
        if pred.sum().item() == 0 and gt.sum().item() == 0:
            return 1.0, 1.0

        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        pixel_f1 = 2 * precision * recall / (precision + recall + epsilon)

        intersection = (pred & gt).sum()
        union = (pred | gt).sum()
        iou = intersection / (union + epsilon)

        return pixel_f1.item(), iou.item()

    if num_classes == 3:
        f1_scores = []
        iou_scores = []
        for cls in range(num_classes):
            if cls == 0:
                pred_cls = pred == 0
                gt_cls = gt == 0
            else:
                pred_cls = (pred == cls) | (pred == 3)
                gt_cls = (gt == cls) | (pred == 3)

            # Handles empty predictions / ground truth
            if pred_cls.sum().item() == 0 and gt_cls.sum().item() == 0:
                f1_scores.append(torch.tensor(1.0).to(pred.device))
                iou_scores.append(torch.tensor(1.0).to(pred.device))
                continue

            tp = (pred_cls & gt_cls).sum().float()
            fp = (pred_cls & ~gt_cls).sum().float()
            fn = (~pred_cls & gt_cls).sum().float()

            epsilon = 1e-7
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)

            intersection = (pred_cls & gt_cls).sum()
            union = (pred_cls | gt_cls).sum()
            iou = intersection / (union + epsilon)

            f1_scores.append(f1)
            iou_scores.append(iou)

        return torch.stack(f1_scores), torch.stack(iou_scores)
        # mean_f1 = torch.mean(torch.stack(f1_scores))
        # mean_iou = torch.mean(torch.stack(iou_scores))
        # return mean_f1.item(), mean_iou.item()

    return 0.0, 0.0


def main():
    # Prepare your ground truth data (manually or load from file)
    # An example ground truth dict might look like this:
    gt_annotations = {
        "images": [
            {
                "id": 1,
                "file_name": "",
                "height": 1024,
                "width": 1024,
            },
            # More images...
        ],
        "annotations": [
            {
                "id": 1,  # Unique id of this annotation
                "category_id": 1,  # Category of the object
                "image_id": 1,
                "bbox": [
                    0,
                    0,
                    200,
                    200,
                ],  # Format of the bbox (top left x, top left y, width, height)
                "area": 40000,
                "iscrowd": 0,
            },
            # More annotations...
        ],
        "categories": [
            {
                "id": 1,
                "supercategory": "building",
                "name": "building",
            },
        ],
    }

    # Prepare your detection results (manually or load from file)
    # An example detections dict might look like this:
    detections = [
        {
            "image_id": 1,
            "category_id": 1,  # Category of the object
            "bbox": [
                10,
                10,
                200,
                200,
            ],  # Format of the bbox (top left x, top left y, width, height)
            "score": 0.6,  # Confidence score for this prediction
        },
    ]
