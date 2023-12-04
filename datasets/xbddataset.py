from os import listdir, path
from json import load
import random

from torch import tensor
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from shapely.wkt import loads

import torchvision.transforms as T
import groundingdino.datasets.transforms as DT

ImageFile.LOAD_TRUNCATED_IMAGES = True
FILE_OVERRIDE = None

# in test
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/test/images/socal-fire_00001252_pre_disaster.png"]
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/test/images/hurricane-harvey_00000360_pre_disaster.png"]  # 0.15
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/test/images/hurricane-harvey_00000424_pre_disaster.png"]  # 0.15
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/test/images/hurricane-michael_00000399_pre_disaster.png"]  # 0.15
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/test/images/hurricane-michael_00000447_pre_disaster.png"]  # 0.15
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/test/images/hurricane-michael_00000478_pre_disaster.png"]  # 0.15
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/test/images/hurricane-michael_00000481_pre_disaster.png"]  # 0.15
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/test/images/palu-tsunami_00000008_pre_disaster.png"]  # 0.15

# in tier3
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/tier3/images/joplin-tornado_00000000_pre_disaster.png"]

# in train
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/train/images/hurricane-harvey_00000233_post_disaster.png"]
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/train/images/hurricane-michael_00000473_pre_disaster.png"]
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/train/images/santa-rosa-wildfire_00000039_pre_disaster.png"]
# FILE_OVERRIDE = ["/home/uq-yiyun/xbd/train/images/hurricane-harvey_00000206_pre_disaster.png"]

class XBDDataset(Dataset):
    """
    Return: dict(
        file_name, image_dim
        pre_image_original, pre_image_mask, pre_label, pre_image
        post_image_original, post_image_mask, post_label, post_image
    )
    """

    def __init__(
        self,
        dataset_paths: list,
        dino_transform=False,
        include_post=False,  # include post-disaster images
        include_masks=False,  # get the mask images of the building generated from preprocess-data.py
        include_json_labels=False,
        include_bbox_labels=True,  # include all bounding boxes of the buildings in the image
    ):
        self.pre_image_paths = []
        if FILE_OVERRIDE is not None and len(FILE_OVERRIDE) > 0:
            self.pre_image_paths = FILE_OVERRIDE
        else:
            for folder in dataset_paths:
                for file in listdir(path.join(folder, "images")):
                    if file.endswith("_pre_disaster.png"):
                        self.pre_image_paths.append(path.join(folder, "images", file))

        self.dino_transform = None
        if dino_transform:
            self.dino_transform = DT.Compose(
                [
                    DT.RandomResize([800], max_size=1333),
                    DT.ToTensor(),
                    DT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        self.include_post = include_post
        self.include_masks = include_masks
        self.include_json_labels = include_json_labels
        self.include_bbox_labels = include_bbox_labels

    def __len__(self):
        return len(self.pre_image_paths)

    def __getitem__(self, idx):
        file_name = self.pre_image_paths[idx]
        return_item = {
            "pre_file_name": file_name,
            "post_file_name": file_name.replace(
                "_pre_disaster.png", "_post_disaster.png"
            ),
        }

        pre_image_original = Image.open(file_name).convert("RGB")
        return_item["pre_image_original"] = T.ToTensor()(pre_image_original)

        width, height = pre_image_original.size
        return_item["image_dim"] = (width, height)

        pre_json_file = file_name.replace("/images/", "/labels/").replace(
            ".png", ".json"
        )

        if self.include_masks:
            return_item["pre_image_mask"] = T.ToTensor()(
                Image.open(file_name.replace("/images/", "/masks/")).convert("L")
            )

        if self.include_bbox_labels:
            pre_json_data = load(open(pre_json_file))["features"]["xy"]
            pre_bounding_boxes = []
            for property in pre_json_data:
                bounds = list(loads(property["wkt"]).bounds)  # (x1, y1, x2, y2)
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                area = width * height

                pre_bounding_boxes.append(
                    {
                        "xywh": tensor([bounds[0], bounds[1], width, height]),
                        "area": area,
                    }
                )
            return_item["pre_bbox_label"] = pre_bounding_boxes

        if self.include_json_labels:
            return_item["pre_label"] = load(open(pre_json_file))["features"]["xy"]

        if self.dino_transform:
            return_item["pre_image"] = self.dino_transform(pre_image_original, None)[0]

        if self.include_post:
            post_image_original = Image.open(
                file_name.replace("_pre_disaster.png", "_post_disaster.png")
            ).convert("RGB")
            return_item["post_image_original"] = T.ToTensor()(post_image_original)

            post_json_file = file_name.replace("/images/", "/labels/").replace(
                "_pre_disaster.png", "_post_disaster.json"
            )

            if self.include_masks:
                return_item["post_image_mask"] = T.ToTensor()(
                    Image.open(
                        file_name.replace("/images/", "/masks/").replace(
                            "_pre_disaster.png", "_post_disaster.png"
                        )
                    ).convert("L")
                )

            if self.include_json_labels:
                return_item["post_label"] = load(open(post_json_file))["features"]["xy"]

            if self.dino_transform:
                return_item["post_image"] = self.dino_transform(
                    post_image_original, None
                )[0]

        return return_item


class XBDBldGTDataset(Dataset):
    """
    Patch of the buildings of xBD dataset (Ground Truth Bounding Box)
    """

    def __init__(
        self,
        dataset_paths: list,
        damage_types: list,  # e.g. ["no-damage", "minor-damage", "major-damage", "destroyed"]
        # include_pre=True,  # include pre-disaster building patch
        # include_post=True,  # include post-disaster building patch of the same building location
        shuffle=True,  # shuffle the order of the building patches
        area_filter=False,  # filter out small buildings
        area_filter_min=5000,  # minimum area of the building to be included
        image_buffer=True,  # buffer the patch to give CLIP more context for prediction
        image_buffer_size=10,  # buffer size
        image_buffer_dim_min=100,  # minimum dimension of the buffered image
        type_sample_limit=False,  # limit the number of samples per damage type
        type_sample_limit_max=1000,  # maximum number of samples per damage type
        max_samples=None,  # maximum number of all samples
        clip_preprocess_fn=None,  # preprocess the images for CLIP, allowing multiple images in a batch
    ):
        self.pre_image_paths = []
        for folder in dataset_paths:
            for file in listdir(path.join(folder, "images")):
                if file.endswith("_pre_disaster.png"):
                    self.pre_image_paths.append(path.join(folder, "images", file))
        self.damage_types = damage_types
        self.type_start_ids = []
        self.all_buildings = dict()
        for type in self.damage_types:
            self.all_buildings[type] = []
        self.clip_preprocess_fn = clip_preprocess_fn

        width, height = Image.open(self.pre_image_paths[0]).convert("RGB").size
        sample_count = 0

        for file in self.pre_image_paths:
            if max_samples is not None and sample_count >= max_samples:
                break

            # read the json file
            pre_json_file = file.replace("/images/", "/labels/").replace(
                ".png", ".json"
            )
            pre_json_data = load(open(pre_json_file))

            post_json_file = file.replace("/images/", "/labels/").replace(
                "_pre_disaster.png", "_post_disaster.json"
            )
            post_json_data = load(open(post_json_file))

            for index, property in enumerate(pre_json_data["features"]["xy"]):
                if max_samples is not None and sample_count >= max_samples:
                    break

                polygon = property["wkt"]

                # Note: This assumes the orders of buildings in pre.json and post.json are the same
                damage_type = post_json_data["features"]["xy"][index]["properties"][
                    "subtype"
                ]

                if damage_type not in self.damage_types:
                    continue

                if type_sample_limit:
                    if len(self.all_buildings[damage_type]) >= type_sample_limit_max:
                        continue

                # find the bounding box of the pre-disaster building
                # give POLYGON(()) string, find the enclosing bounding box,
                # return a tuple of (x1, y1, x2, y2)

                # split the string into a list of points
                points = polygon[10:-2].split(", ")
                # convert the list of points into a list of tuples
                points = [tuple(map(float, point.split(" "))) for point in points]
                # find the min and max x and y values
                x1 = min(points, key=lambda x: x[0])[0]
                y1 = min(points, key=lambda x: x[1])[1]
                x2 = max(points, key=lambda x: x[0])[0]
                y2 = max(points, key=lambda x: x[1])[1]

                w = x2 - x1
                h = y2 - y1
                area = w * h

                if area_filter and area < area_filter_min:
                    continue

                if image_buffer:
                    image_buffer_x = (
                        (int((image_buffer_dim_min - w) / 2.0) + image_buffer_size)
                        if w < image_buffer_dim_min
                        else image_buffer_size
                    )
                    image_buffer_y = (
                        (int((image_buffer_dim_min - h) / 2.0) + image_buffer_size)
                        if h < image_buffer_dim_min
                        else image_buffer_size
                    )

                    # Add padding for prediction
                    x1_pad = max(int(round(x1 - image_buffer_x)), 0)
                    y1_pad = max(int(round(y1 - image_buffer_y)), 0)
                    x2_pad = min(int(round(x2 + image_buffer_x)), width)
                    y2_pad = min(int(round(y2 + image_buffer_y)), height)

                    self.all_buildings[damage_type].append(
                        {
                            "pre-file": file,
                            "post-file": file.replace(
                                "_pre_disaster.png", "_post_disaster.png"
                            ),
                            "bbox": (x1_pad, y1_pad, x2_pad, y2_pad),
                        }
                    )

                    sample_count += 1

        print(f"Total number of building patches: {len(self)}")
        for idx, type in enumerate(self.damage_types):
            if shuffle:
                random.shuffle(self.all_buildings[type])
            start_idx = 0
            if idx == 0:
                self.type_start_ids.append(start_idx)
            else:
                start_idx = self.type_start_ids[idx - 1] + len(
                    self.all_buildings[self.damage_types[idx - 1]]
                )
                self.type_start_ids.append(start_idx)
            print(f"{type}: {len(self.all_buildings[type])} ({start_idx})")

    def __len__(self):
        count = 0
        for type in self.damage_types:
            count += len(self.all_buildings[type])
        return count

    def __getitem__(self, idx):
        match_type = None
        pos = 0
        for index, type in enumerate(self.damage_types):
            if idx >= self.type_start_ids[index]:
                match_type = type
                pos = idx - self.type_start_ids[index]
            else:
                break

        # read the image
        pre_file_name = self.all_buildings[match_type][pos]["pre-file"]
        post_file_name = self.all_buildings[match_type][pos]["post-file"]
        pre_image = Image.open(pre_file_name).convert("RGB")
        post_image = Image.open(post_file_name).convert("RGB")

        # crop the image
        x1, y1, x2, y2 = self.all_buildings[match_type][pos]["bbox"]
        pre_image = pre_image.crop((x1, y1, x2, y2))
        post_image = post_image.crop((x1, y1, x2, y2))

        if self.clip_preprocess_fn is not None:
            pre_image = self.clip_preprocess_fn(pre_image)
            post_image = self.clip_preprocess_fn(post_image)
        else:
            # convert to tensor
            pre_image = T.ToTensor()(pre_image)
            post_image = T.ToTensor()(post_image)

        # pre_image = DT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(pre_image, None)[0]
        # post_image = DT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(post_image, None)[0]

        return {
            "pre_file_name": pre_file_name,
            "post_file_name": post_file_name,
            "bbox": tensor([x1, y1, x2, y2]),
            "pre_patch": pre_image,
            "post_patch": post_image,
            "damage_type": match_type,
        }


class XBDBldLocPLDataset(Dataset):
    """
    Read the pseudo-labels from stage 1 (building bboxes) from the json file.
    """

    def __init__(
        self,
        json_file,
        clip_preprocess_fn=None,  # preprocess the images for CLIP
        image_buffer=True,  # buffer the patch to give CLIP more context for prediction
        image_buffer_size=10,  # buffer size
        image_buffer_dim_min=100,  # minimum dimension of the buffered image
        image_width=1024,
    ):
        # read json file
        self.json_data = load(open(json_file))
        self.clip_preprocess_fn = clip_preprocess_fn

        for building in self.json_data:
            x1, y1, x2, y2 = [i * image_width for i in building["bbox"]]

            # Apply buffer to the building
            if image_buffer:
                w = x2 - x1
                h = y2 - y1

                image_buffer_x = (
                    (int((image_buffer_dim_min - w) / 2.0) + image_buffer_size)
                    if w < image_buffer_dim_min
                    else image_buffer_size
                )
                image_buffer_y = (
                    (int((image_buffer_dim_min - h) / 2.0) + image_buffer_size)
                    if h < image_buffer_dim_min
                    else image_buffer_size
                )

                # Add padding for prediction
                x1_pad = max(int(round(x1 - image_buffer_x)), 0)
                y1_pad = max(int(round(y1 - image_buffer_y)), 0)
                x2_pad = min(int(round(x2 + image_buffer_x)), image_width)
                y2_pad = min(int(round(y2 + image_buffer_y)), image_width)

                building["bbox"] = (x1_pad, y1_pad, x2_pad, y2_pad)

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        # read the image
        building = self.json_data[idx]
        post_image = Image.open(
            building["pre_image"].replace("_pre_disaster.png", "_post_disaster.png")
        ).convert("RGB")

        # crop the image
        x1, y1, x2, y2 = building["bbox"]
        post_image = post_image.crop((x1, y1, x2, y2))

        if self.clip_preprocess_fn is not None:
            post_image = self.clip_preprocess_fn(post_image)
        else:
            # convert to tensor
            post_image = T.ToTensor()(post_image)

        return {
            "pre_file_name": building["pre_image"],
            "post_file_name": building["pre_image"].replace(
                "_pre_disaster.png", "_post_disaster.png"
            ),
            "bbox": tensor([x1, y1, x2, y2]),
            "post_patch": post_image,
            "score": building["score"],
        }


class XBDBldClsPLDataset(Dataset):
    """
    Read the pseudo-labels (damage detection) from the json file.
    """

    def __init__(self, json_file, image_dir, preprocess):
        # read json file
        self.json_data = load(open(json_file))
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        # read the image

        if self.image_dir != "":
            post_file_name = path.join(
                self.image_dir, self.json_data[idx]["post_file_name"]
            )
        else:
            # fixme
            post_file_name = self.json_data[idx]["post_file_name"][0]
            # post_file_name = self.json_data[idx]["post_file_name"]
        post_image = Image.open(post_file_name).convert("RGB")

        # crop the image
        x1, y1, x2, y2 = self.json_data[idx]["bbox"]
        post_image = post_image.crop((x1, y1, x2, y2))

        # save image
        # type = self.json_data[idx]["damage_type"]
        # post_image.save(f'test/{type}-{post_file_name.split("/")[-1]}.png')

        post_image = self.preprocess(post_image)

        return {
            "post_file_name": post_file_name,
            "post_patch": post_image,
            "damage_type": self.json_data[idx]["damage_type"],
        }


def dataset_statistics():
    pass


if __name__ == "__main__":
    dataset_statistics()
