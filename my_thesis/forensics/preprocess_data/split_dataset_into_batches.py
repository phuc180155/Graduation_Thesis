import os, sys
import os.path as osp
sys.path.append(osp.dirname(__file__))

from glob import glob
from tqdm import tqdm
import random
import shutil
import math

random.seed(0)

def split_directory_into_batches(src_dir="../../../../2_Deep_Learning/Dataset/facial_forgery/df_in_the_wild/images/test/0_real",\
                                 batch_size=500,\
                                 root_dir="../../../../2_Deep_Learning/Dataset/facial_forgery/df_in_the_wild/image"):
    phase = src_dir.split("/")[-2]
    type = src_dir.split("/")[-1]
    # Destination directory
    dst_dir = osp.join(root_dir, phase, type)

    # Images:
    images = os.listdir(src_dir)
    num_img = len(images)
    num_subfolder = math.ceil(1.0 * num_img / batch_size)

    print("\n=============================")
    print("Src dir: ", src_dir)
    print("Number of images: ", num_img)
    print("Number of subfolder: ", num_subfolder)

    for i in tqdm(range(num_subfolder)):
        dst = osp.join(dst_dir, str(i))
        if not osp.exists(dst):
            os.mkdir(dst)
        for j in range(batch_size * i, min(batch_size * i + batch_size, num_img)):
            img_name = images[j]
            img_path = osp.join(src_dir, img_name)
            shutil.move(img_path, dst)

def split_dataset_into_batches(dataset="dfdc"):
    # train:
    src_dir = "../../../../2_Deep_Learning/Dataset/facial_forgery/{}/images".format(dataset)
    root_dir = "../../../../2_Deep_Learning/Dataset/facial_forgery/{}/image".format(dataset)

    for phase in ['train', 'test', 'val']:
        for type in ['0_real', '1_df']:
            src = "{}/{}/{}".format(src_dir, phase, type)
            split_directory_into_batches(src_dir=src, batch_size=500, root_dir=root_dir)
    print("!Done dataset {}".format(dataset))
        
if __name__ == '__main__':
    split_dataset_into_batches(dataset="df_in_the_wild")

