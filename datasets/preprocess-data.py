"""Create masks for xBD dataset"""
import numpy as np
from os import path, makedirs, listdir
from multiprocessing import Pool
import json
import cv2
from shapely.wkt import loads
import time
import argparse
from tqdm import tqdm


IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024


damage_dict = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 5,
}

# optional: for visualisation only
visualize_mask = False
damage_dict_bgr = {
    "no-damage": (70, 172, 0),
    "minor-damage": (0, 197, 253),
    "major-damage": (0, 140, 253),
    "destroyed": (0, 0, 220),
    "un-classified": (255, 255, 255),
}

# Binary classification
damage_dict_bgr_2 = {
    "no-damage": (70, 172, 0),
    "minor-damage": (0, 140, 253),
    "major-damage": (0, 140, 253),
    "destroyed": (0, 140, 253),
    "un-classified": (70, 172, 0),
}


def mask_for_polygon(poly, im_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    img_mask = np.zeros(im_size, np.uint8)

    def int_coords(x):
        return np.array(x).round().astype(np.int32)

    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def process_image(json_file: str):
    pre_disaster_label = json.load(open(json_file))
    post_disaster_label = json.load(
        open(json_file.replace("_pre_disaster", "_post_disaster"))
    )

    building_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    damage_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)

    for property in pre_disaster_label["features"]["xy"]:
        polygon = loads(property["wkt"])
        building_mask[mask_for_polygon(polygon) > 0] = 255

    # fixme: use pre or post?
    for property in post_disaster_label["features"]["xy"]:
        polygon = loads(property["wkt"])
        damage_mask[mask_for_polygon(polygon) > 0] = damage_dict[
            property["properties"]["subtype"]
        ]

    cv2.imwrite(
        json_file.replace("/labels/", "/masks/").replace(
            "_pre_disaster.json", "_pre_disaster.png"
        ),
        building_mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    cv2.imwrite(
        json_file.replace("/labels/", "/masks/").replace(
            "_pre_disaster.json", "_post_disaster.png"
        ),
        damage_mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )

    if visualize_mask:
        # optional: for visualisation only
        damage_mask_bgr = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        for property in post_disaster_label["features"]["xy"]:
            polygon = loads(property["wkt"])
            damage_mask_bgr[mask_for_polygon(polygon) > 0] = damage_dict_bgr[
                property["properties"]["subtype"]
            ]
        cv2.imwrite(
            json_file.replace("/labels/", "/masks/").replace(
                "_pre_disaster.json", "_post_disaster_rgb.png"
            ),
            damage_mask_bgr,
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess data for xBD dataset")
    # Change the path to the location of the data (each contains 'images' and 'labels' folders)
    parser.add_argument(
        "--training-set-path",
        "-tsp",
        type=str,
        required=True,
        help="Path to the training set",
        dest="training_set_path",
    )
    parser.add_argument(
        "--validation-set-path",
        "-vsp",
        type=str,
        required=True,
        help="Path to the validation set",
        dest="validation_set_path",
    )
    parser.add_argument(
        "--test-set-path",
        "-tssp",
        type=str,
        required=True,
        help="Path to the test set",
        dest="test_set_path",
    )
    # Additional tier3 data
    parser.add_argument(
        "--additional-data-path",
        "-adp",
        type=str,
        required=True,
        help="Path to the additional data",
        dest="additional_data_path",
    )

    parser.add_argument(
        "--tiff",
        "-t",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the data is in tiff format",
        dest="tiff",
    )
    parser.add_argument(
        "--num-processes",
        "-np",
        type=int,
        default=8,
        help="Number of processes to use",
        dest="num_processes",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to visualize the masks",
        dest="visualize",
    )

    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()

    args = get_args()
    visualize_mask = args.visualize

    # Gather all pre-disaster labels
    all_files = []
    for directory in [
        args.training_set_path,
        args.validation_set_path,
        args.test_set_path,
        args.additional_data_path,
    ]:
        # make masks directory
        makedirs(path.join(directory, "masks"), exist_ok=True)
        for file in sorted(listdir(path.join(directory, "images"))):
            if args.tiff:
                # check if file ends with _pre_disaster.tif
                if "_pre_disaster.tif" in file:
                    all_files.append(
                        path.join(directory, "labels", file).replace(
                            "_pre_disaster.tif", "_pre_disaster.json"
                        )
                    )
            else:
                # check if file ends with _pre_disaster.png
                if "_pre_disaster.png" in file:
                    all_files.append(
                        path.join(directory, "labels", file).replace(
                            "_pre_disaster.png", "_pre_disaster.json"
                        )
                    )

    print(len(all_files))

    # Multiprocessing
    with Pool(args.num_processes) as p:
        _ = list(tqdm(p.imap(process_image, all_files), total=len(all_files)))
        # p.map(process_image, all_files)

    # without multiprocessing
    # for file in all_files:
    #     process_image(file)

    # Show time in minutes and seconds
    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Time taken: {int(minutes):02d}:{seconds:05.2f}")
