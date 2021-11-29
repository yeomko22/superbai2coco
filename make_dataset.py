import os
import random
import shutil
from collections import defaultdict

random.seed(1234)
bucket_size = 100
bucket_num = 16
classnames = ["chart", "graph", "image", "line", "other", "polygon", "table"]

origin_dir = "/Users/riiid/Downloads/categorized_1024"
filepath_dict = defaultdict(list)
for classname in classnames:
    filepath_generator = os.walk(f"{origin_dir}/{classname}/photo")
    for dir, _, filenames in filepath_generator:
        for filename in filenames:
            ext = filename.split(".")[-1]
            if ext in ("jpg", "jpeg", "JPG", "JPEG", "png", "PNG"):
                filepath_dict[classname].append(f"{dir}/{filename}")

for key in filepath_dict:
    random.shuffle(filepath_dict[key])

dataset_name = "./dataset"
os.mkdir(dataset_name)
for i in range(bucket_num):
    bucket_name = f"bucket_{i}"
    os.mkdir(f"{dataset_name}/bucket_{i}")
    for key in filepath_dict:
        savedir = f"{dataset_name}/{bucket_name}/{key}"
        os.mkdir(savedir)
        for j in range(bucket_size):
            shutil.copyfile(filepath_dict[key][i * bucket_size + j], f"{savedir}/{str(i * bucket_size + j).zfill(4)}.jpg")
