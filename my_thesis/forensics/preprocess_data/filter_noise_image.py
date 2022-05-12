from numpy import void
import face_recognition

import os, sys
from os.path import join
from glob import glob
from tqdm import tqdm
import shutil
import argparse

def inspect_facial_image(img_path: str) -> bool:
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)
    is_face = True if len(face_locations) else False
    return is_face
    
def filter(dataset_path: str)->void:
    train_set = join(dataset_path, 'train')
    test_set = join(dataset_path, 'test')
    val_set = join(dataset_path, 'val')
    dsets = [train_set, test_set, val_set]
    
    noise_dir = join(dataset_path, 'noise')
    log_txt = open(join(dataset_path, 'log_noise.txt'), 'w')
    
    cnt = 0
    for dset in dsets:
        print(dset)
        img_paths = glob(join(dset, '*/*'))
        for img_path in tqdm(img_paths):
            if inspect_facial_image(img_path):
                continue
            cnt += 1
            log_txt.write(img_path + '\n')
            shutil.copy(img_path, noise_dir)
    print("Number of noise image: ", cnt)
            
def parse_args():
    parser = argparse.ArgumentParser(description="Filter noise image by another face detection module")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to dataset")
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    filter(str(args.dataset_path))
        
    
    
    
    
