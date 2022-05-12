from ast import parse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import argparse
import glob
import os.path as osp
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from utils import *
import torch
import os

torch.multiprocessing.set_start_method('spawn')
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
face_detector = MTCNN(device=device)

fake_image_dir = "../../../../2_Deep_Learning/Dataset/facial_forgery/FF+/image/fake"
fake_video_dir = "../../../../2_Deep_Learning/Dataset/facial_forgery/FF+/manipulated_sequences"
real_image_dir = "../../../../2_Deep_Learning/Dataset/facial_forgery/FF+/image/real"
real_video_dir = "../../../../2_Deep_Learning/Dataset/facial_forgery/FF+/original_sequences"


def parse_args():
    parser = argparse.ArgumentParser(description="Fake Detection")
    parser.add_argument('--in_dir', default='', help='path to train data')
    parser.add_argument('--out_dir', default='', help='path to test data')
    parser.add_argument('--num_thread', default=4, type=int, help='number of threads')
    parser.add_argument('--duration', default=15, type=int)
    return parser.parse_args()

def extract_frame(video_path):
    # 
    output_dir = args.out_dir
    duration = args.duration

    # Đọc video với videoCapture
    video = cv2.VideoCapture(video_path)
    video_name = osp.basename(video_path).split('.')[0]

    print(video_name)
    print(video)

    success = True
    image = None
    id_frame = 0
    
    while success:
        # Đọc video 4 lần (4 frame), chỉ lấy frame cuối cùng, nếu fail thì kết thúc vòng lặp
        for i in range(duration):
            success, image = video.read()
            if not success:
                break
            
        print("Get here")
        print("Success: ", success)
        print("Image: \n", image)
        try:
            # Convert from BGR (CV) to RGB (Image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            continue

        plt.imsave(osp.join(output_dir, video_name + '_' + str(id_frame) + '.jpg'), image, format='jpg')
        id_frame += 1
        

def extract_face(video_path: str, ext_margin=0.2):
    #
    # forgery_tech = video_path.split('/')[-4]
    output_dir = args.out_dir if args.out_dir != '' else real_image_dir
    duration = args.duration
    # Read video:

    video = cv2.VideoCapture(video_path)
    video_name = osp.basename(video_path).split('.')[0]

    # print("Forgery Tech: ", forgery_tech)
    print("Video name: ", video_name)
    # if not osp.exists(osp.join(output_dir, forgery_tech)):
    #     os.mkdir(osp.join(output_dir, forgery_tech))
    #
    success = True
    image = None
    id_frame = 0
    while success:
        # Read <duration> times, only get the last success if successes for all <duration> times
        for _ in range(duration):
            success, image = video.read()
            if not success:
                break
        try:
            # Convert from BGR to RGB
            image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
        except:
            continue

        # Detect face:
        face_pos, probs = face_detector.detect(image, landmarks=False) 
        # vis_face_img(image, face_pos[0], probs[0], landmark=None, vis=False, save=None)

        # Find the face with largest prob 
        try:
            face_pos = face_pos[np.argmax(probs)]
        except:
            continue
        
        # Extend face rectangle
        xmin, ymin, xmax, ymax = face_pos
        x, y, w, h = map(int, [xmin, ymin, xmax-xmin, ymax-ymin])
        extend_x, extend_y = int(w * ext_margin), int(h * ext_margin)
        xmin = max(0, x - extend_x)
        ymin = max(0, y - extend_y)
        xmax = min(image.shape[1], x + w + extend_x)
        ymax = min(image.shape[0], y + h + extend_y)

        face = image[ymin:ymax, xmin:xmax]
        # Save to output dir:
        # plt.imsave(osp.join(output_dir, forgery_tech, video_name + '_' + str(id_frame) + '.jpg'), face, format='jpg')
        plt.imsave(osp.join(output_dir, video_name + '_' + str(id_frame) + '.jpg'), face, format='jpg')

        id_frame += 1
    print("Number of taken frame: {}\n".format(id_frame))


args = parse_args()
if __name__ == '__main__':
    video_paths = []
    video_types = ['/*/*/*/*.mp4', '/*/*/*/*.avi']  # Deepfakes/c23/videos/*.mp4
    in_dir = args.in_dir if args.in_dir != '' else real_video_dir
    # Duyệt tất cả các path tới video
    for type in video_types:
        paths = glob.glob(in_dir + type)
        video_paths.extend(paths)

    video_paths = [p.replace("\\", "/") for p in video_paths]
    print("Paths: ", len(video_paths))

    # Sử dụng tính toán đa luồng
    for path in video_paths:
        extract_face(path)
    