import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List
from PIL import Image

from tqdm import tqdm

random.seed("1234")


def get_categories(superbai_path: str, merge_label: bool) -> (List[dict], dict):
    categories = []
    category_label_to_id = {}
    with open(f"{superbai_path}/project.json") as f:
        project_json = json.loads(f.read())
        object_classes = project_json["object_detection"]["object_classes"]
        for i, object_class in enumerate(object_classes):
            categories.append({
                "supercategory": "none",
                "id": i,
                "name": object_class["name"]
            })
            category_label_to_id[object_class["id"]] = i
    if merge_label:
        categories = [categories[0]]
        categories[0]["name"] = "figure"
        category_label_to_id = {key: 0 for key in category_label_to_id}
    return categories, category_label_to_id


def get_origin_image_paths(origin_dir:str, superbai_dir: str) -> (List[str], dict):
    origin_image_paths = []
    origin_path_to_label = {}
    filepath_generator = os.walk(f"{superbai_dir}")
    for directory, _, filenames in filepath_generator:
        for filename in filenames:
            if ".json" not in filename:
                continue
            with open(f"{directory}/{filename}") as f:
                meta_json = json.loads(f.read())
                origin_path = f"{origin_dir}/{meta_json['data_key']}"
                origin_image_paths.append(origin_path)
                origin_path_to_label[origin_path] = meta_json["label_id"]
    return origin_image_paths, origin_path_to_label


def get_dst_paths(output_dir: str, dirtype: str, src_paths: List[str], origin_path_to_label: dict) -> List[str]:
    dst_paths = []
    for src_path in src_paths:
        dst_paths.append(f"{output_dir}/{dirtype}/{origin_path_to_label[src_path]}.jpg")
    return dst_paths


def copy_files(src_paths: List[str], dst_paths: List[str]) -> None:
    print(f"start copying {len(src_paths)} files")
    for src_path, dst_path in tqdm(zip(src_paths, dst_paths)):
        shutil.copyfile(src_path, dst_path)
    print(f"finish copying {len(src_paths)} files")


def split(train_origin_dir: str, val_origin_dir, superbai_dir: str,
          superbai_train_meta: str, superbai_val_meta: str, output_dir: str) -> (List[str], List[str]):
    # split train, val data
    src_train_paths, train_origin_path_to_label = get_origin_image_paths(
        origin_dir=train_origin_dir,
        superbai_dir=f"{superbai_dir}/meta/{superbai_train_meta}"
    )
    src_val_paths, val_origin_path_to_label = get_origin_image_paths(
        origin_dir=val_origin_dir,
        superbai_dir=f"{superbai_dir}/meta/{superbai_val_meta}"
    )

    # copy train images to output dir
    train_labels = []
    dst_train_paths = []
    for src_train_path in src_train_paths:
        train_label = train_origin_path_to_label[src_train_path]
        train_labels.append(train_label)
        dst_train_paths.append(f"{output_dir}/train/{train_label}.jpg")
    copy_files(src_train_paths, dst_train_paths)

    # copy val images to output dir
    val_labels = []
    dst_val_paths = []
    for src_val_path in src_val_paths:
        val_label = val_origin_path_to_label[src_val_path]
        val_labels.append(val_label)
        dst_val_paths.append(f"{output_dir}/val/{val_label}.jpg")
    copy_files(src_val_paths, dst_val_paths)
    return train_labels, val_labels


def get_image_info_dict(superbai_dir: str) -> dict:
    """
    :param superbai_dir:
    :return: dictionary
        key: hash value created by superbai
        value: image widht, height
    """
    image_info_dict = {}
    filepath_generator = os.walk(f"{superbai_dir}/meta")
    for directory, _, filenames in filepath_generator:
        for filename in filenames:
            if ".json" not in filename:
                continue
            with open(f"{directory}/{filename}") as f:
                meta_json = json.loads(f.read())
                image_info_dict[meta_json["label_id"]] = {
                    "width": meta_json["image_info"]["width"],
                    "height": meta_json["image_info"]["height"],
                }
    return image_info_dict


def get_bbox_dict(superbai_dir: str, category_label_to_id: dict) -> dict:
    """
    :param superbai_dir:
    :param category_label_to_id:
    :return: dictionary
        key: hash value created by superbai
        value: bounding box coords and label id
    """
    labels_dir = f"{superbai_dir}/labels"
    filenames = os.listdir(labels_dir)
    bbox_dict = defaultdict(list)
    for filename in filenames:
        label = filename.replace(".json", "")
        with open(f"{labels_dir}/{filename}") as f:
            label_json = json.loads(f.read())
            objects = label_json["objects"]
            for object in objects:
                bbox_dict[label].append({
                    "bbox": [
                        object["annotation"]["coord"]["x"],
                        object["annotation"]["coord"]["y"],
                        object["annotation"]["coord"]["width"],
                        object["annotation"]["coord"]["height"],
                    ],
                    "class_id": category_label_to_id[object["class_id"]],
                })
    return bbox_dict


def add_padding(output_dir, target_dir, labels, bbox_dict):
    for label in labels:
        imagepath = f"{output_dir}/{target_dir}/{label}.jpg"
        image = Image.open(imagepath).convert("RGB")
        width, height = image.size
        if width == height:
            continue
        elif width > height:
            padded = Image.new(image.mode, (width, width), 0)
            move = (width - height) // 2
            padded.paste(image, (0, move))
            for bboxes in bbox_dict[label]:
                for bbox_d in bboxes:
                    bbox_d["bbox"][1] + move
        else:
            padded = Image.new(image.mode, (height, height), 0)
            move = (height - width) // 2
            padded.paste(image, (move, 0))
            for bboxes in bbox_dict[label]:
                for bbox_d in bboxes:
                    bbox_d["bbox"][0] + move
        padded.save(imagepath)


def get_coco_json(labels: List[str], categories: List[dict], image_info_dict: dict, bbox_dict: dict):
    image_id = 1
    bbox_id = 1
    coco_json = {
        "type": "instances",
        "categories": categories,
        "images": [],
        "annotations": []
    }
    for label in labels:
        coco_json["images"].append({
            "file_name": f"{label}.jpg",
            "height": image_info_dict[label]["height"],
            "width": image_info_dict[label]["width"],
            "id": image_id,
        })
        bboxes = bbox_dict[label]
        for bbox in bboxes:
            coco_json["annotations"].append({
                'image_id': image_id,
                'id': bbox_id,
                'area': image_info_dict[label]["height"] * image_info_dict[label]["width"],
                'iscrowd': 0,
                'bbox': [int(x) for x in bbox["bbox"]],
                'category_id': bbox["class_id"],
                'ignore': 0,
                'segmentation': []  # This script is not for segmentation
            })
            bbox_id += 1
        image_id += 1
    return coco_json


parser = argparse.ArgumentParser(
    description='This script support converting voc format xmls to coco format json')
parser.add_argument('--superbai_dir', required=True, type=str, help='path to superbai directory.')
parser.add_argument('--superbai_train_meta', required=True, type=str, help='path to superbai directory.')
parser.add_argument('--superbai_val_meta', required=True, type=str, help='path to superbai directory.')
parser.add_argument('--train_origin', required=True, type=str, help='path to train origin dataset.')
parser.add_argument('--val_origin', required=True, type=str, help='path to val origin dataset.')
parser.add_argument('--output_dir', required=True, type=str, help='path to output json file')
parser.add_argument('--merge_label', default=True, type=bool, help='whether merge the labels into one label')
parser.add_argument('--padding', default=True, type=bool, help='whether to add padding')


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    superbai_dir = args.superbai_dir
    superbai_train_meta = args.superbai_train_meta
    superbai_val_meta = args.superbai_val_meta
    train_origin = args.train_origin
    val_origin = args.val_origin
    output_dir = args.output_dir
    merge_label = args.merge_label
    padding = args.padding


    # generate output directories
    Path(f"{output_dir}/annotations").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/train").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/val").mkdir(parents=True, exist_ok=True)

    # preprocess superbai style labels to generte coco json
    categories, category_label_to_id = get_categories(superbai_dir, merge_label)

    image_info_dict = get_image_info_dict(superbai_dir)
    bbox_dict = get_bbox_dict(superbai_dir, category_label_to_id)

    # split  train, val dataset
    train_labels, val_labels = split(
        train_origin_dir=train_origin,
        val_origin_dir=val_origin,
        superbai_dir=superbai_dir,
        superbai_train_meta=superbai_train_meta,
        superbai_val_meta=superbai_val_meta,
        output_dir=output_dir
    )
    if padding:
        add_padding(output_dir, "train", train_labels, bbox_dict)
        add_padding(output_dir, "val", val_labels, bbox_dict)

    # generate and write coco json
    train_coco_json = get_coco_json(train_labels, categories, image_info_dict, bbox_dict)
    val_coco_json = get_coco_json(val_labels, categories, image_info_dict, bbox_dict)
    with open(f"{output_dir}/annotations/custom_train.json", "w") as f:
        f.write(json.dumps(train_coco_json))
    with open(f"{output_dir}/annotations/custom_val.json", "w") as f:
        f.write(json.dumps(val_coco_json))
