import boto3
import time
import os
from tqdm import tqdm

s3 = boto3.resource("s3")


def get_file_cnt(bucket_name, s3_folder):
    bucket = s3.Bucket(bucket_name)
    filename_set = set()
    totalCount = 0
    start = time.time()
    for obj in bucket.objects.filter(Prefix=s3_folder).all():
        print(obj.key)
        if obj.key in filename_set:
            print("duplicate!", len(filename_set), obj.key)
            break
        filename_set.add(obj.key)
        if not (totalCount + 1) % 1000:
            print("counting files", totalCount + 1)
        totalCount += 1
    print("totalCount: ", totalCount, "elapsed: ", time.time() - start)
    return totalCount


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    # total_cnt = get_file_cnt(bucket_name, s3_folder)
    total_cnt = 10000
    bucket = s3.Bucket(bucket_name)
    cnt = 0
    with tqdm(total=total_cnt) as pbar:
        for obj in bucket.objects.filter(Prefix=s3_folder):
            if cnt > total_cnt:
                break
            target = f"{local_dir}/{str(cnt).zfill(6)}.jpg"
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == '/':
                continue
            bucket.download_file(obj.key, target)
            cnt += 1
            pbar.update(1)
get_file_cnt("riiid-rb2a-qna-data-prod", "app-server-student/2021/11/1/")

# download_s3_folder("riiid-rb2a-qna-data-prod", "app-server-student/2021/11/1", "livedata/11/1")
