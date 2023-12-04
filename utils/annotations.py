import numpy as np
import torch
from typing import List
from torchvision.ops import box_convert
import supervision as sv
from supervision.draw.color import Color
import cv2

# Reference: Grounding DINO
def annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: List[str],
) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]

    box_annotator = sv.BoxAnnotator(color=Color.white())
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )
    return annotated_frame


def gdino_visualize(source_image, file_name, boxes, logits, phrases):
    annotated_frame = annotate(
        image_source=source_image,
        boxes=boxes,
        logits=logits,
        phrases=phrases,
    )
    cv2.imwrite(file_name, annotated_frame)
