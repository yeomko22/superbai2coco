import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List

from tqdm import tqdm

random.seed("1234")


def get_categories(superbai_path: str) -> (List[dict], dict):
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
    return categories, category_label_to_id


def get_origin_image_paths(origin_dir:str, superbai_dir: str) -> (List[str], dict):
    origin_image_paths = []
    origin_path_to_label = {}
    filepath_generator = os.walk(f"{superbai_dir}/meta")
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


def split_train_val(origin_paths: List[str], train_ratio: float=0.8) -> (List[str], List[str]):
    random.shuffle(origin_paths)
    train_size = int(len(origin_paths) * train_ratio)
    src_train_paths = origin_paths[:train_size]
    src_val_paths = origin_paths[train_size:]
    return src_train_paths, src_val_paths


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


def split(origin_dir: str, superbai_dir: str, output_dir: str) -> (List[str], List[str]):
    # split train, val data
    origin_image_paths, origin_path_to_label = get_origin_image_paths(origin_dir, superbai_dir)
    src_train_paths, src_val_paths = split_train_val(origin_image_paths)

    # copy train images to output dir
    train_labels = []
    dst_train_paths = []
    for src_train_path in src_train_paths:
        train_label = origin_path_to_label[src_train_path]
        train_labels.append(train_label)
        dst_train_paths.append(f"{output_dir}/train/{train_label}.jpg")
    copy_files(src_train_paths, dst_train_paths)

    # copy val images to output dir
    val_labels = []
    dst_val_paths = []
    for src_val_path in src_val_paths:
        val_label = origin_path_to_label[src_val_path]
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


def get_coco_json(labels: List[str], categories: List[dict], image_info_dict: dict, bbox_dict: dict):
    image_id = 0
    bbox_id = 0
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
        image_id += 1
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
    return coco_json


parser = argparse.ArgumentParser(
    description='This script support converting voc format xmls to coco format json')
parser.add_argument('--superbai_dir', type=str, default=None, help='path to superbai directory.')
parser.add_argument('--origin_dir', type=str, default=None, help='path to origin dataset.')
parser.add_argument('--output_dir', type=str, default='output.json', help='path to output json file')

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    superbai_dir = args.superbai_dir
    origin_dir = args.origin_dir
    output_dir = args.output_dir

    # generate output directories
    Path(f"{output_dir}/annotations").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/train").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/val").mkdir(parents=True, exist_ok=True)

    # preprocess superbai style labels to generte coco json
    categories, category_label_to_id = get_categories(superbai_dir)
    image_info_dict = get_image_info_dict(superbai_dir)
    bbox_dict = get_bbox_dict(superbai_dir, category_label_to_id)

    # split  train, val dataset
    train_labels, val_labels = split(origin_dir, superbai_dir, output_dir)

    # generate and write coco json
    train_coco_json = get_coco_json(train_labels, categories, image_info_dict, bbox_dict)
    val_coco_json = get_coco_json(train_labels, categories, image_info_dict, bbox_dict)
    with open(f"{output_dir}/annotations/train.json", "w") as f:
        f.write(json.dumps(train_coco_json))
    with open(f"{output_dir}/annotations/val.json", "w") as f:
        f.write(json.dumps(val_coco_json))
