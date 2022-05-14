import os, sys
import os.path as osp
from os.path import join
from tracemalloc import Statistic
from numpy import void

from sklearn.metrics import d2_tweedie_score
root_dir = osp.dirname(__file__)

from glob import glob
from tqdm import tqdm
import random
import shutil
from typing import Any, List, Dict
import argparse
import json

random.seed(0)

def log_dataset_statistic(dataset_path: str, dataset_name: str, statistic_dir: str, device_name: str):
    """Function to save all the image paths of a dataset to a statistic directory

    Args:
        dataset_path (str): path to folder contains dataset, like: ".../image/"
        dataset_name (str): name of dataset
        statistic_dir (str): path to statistic directory
        device_name (str): the name of this device
    """
    dsets = [join(dataset_path, 'train'), join(dataset_path, 'test'), join(dataset_path, 'val')]
    for dset in dsets:
        print(dset)
        with open(join(statistic_dir, "{}_{}_{}.txt".format(device_name, dataset_name, osp.basename(dset))), 'w') as f:
            img_paths = glob(join(dset, "*/*"))
            for img_path in tqdm(img_paths):
                info = img_path.split("/")
                img_name = info[-1]
                cls = info[-2]
                phase = info[-3]
                saved_path = "{}/{}\n".format(cls, img_name)
                f.write(saved_path)
            f.close()
    

def statisticize_dataset(dataset_path: str):
    """ Function to display statistic of a dataset

    Args:
        dataset_path (str): path to folder contains dataset, like: .../image/
    Return:
        Dict[str, Dict[str, int]]
    """

    train_set = join(dataset_path, 'train')
    test_set = join(dataset_path, 'test')
    val_set = join(dataset_path, 'val')
    
    dsets = [train_set, test_set, val_set]

    statistic = {
        train_set: {},
        test_set: {},
        val_set: {}
    }
    for dset in dsets:
        for cls in ['0_real', '1_df', '1_f2f', '1_fs', '1_nt', '1_fake']:
            if not osp.exists(join(dset, cls)):
                continue
            num_samples = len(os.listdir(join(dset, cls)))
            if num_samples:
                statistic[dset][cls] = num_samples
    print(json.dumps(statistic, indent=4))
    return statistic

def get_image_from_txt_file(txt_file: str, head_path: str)->void:
    """
    """
    dset = []
    if not osp.exists(txt_file):
        return None
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            dset.append(head_path + '/' + line)
    return dset
    
def check_synchronization_between_servers(dataset_path: str, train_set: List[str], val_set: List[str], test_set: List[str]) -> bool:
    cur_train_set = glob(join(dataset_path, 'train', '*/*'))
    cur_test_set = glob(join(dataset_path, 'test', '*/*'))
    cur_sets = [cur_train_set, cur_test_set]
    
    txt_train_set = train_set + val_set
    txt_test_set = test_set
    txt_sets = [txt_train_set, txt_test_set]
    
    assert len(txt_train_set) == len(cur_train_set), 'Correct! Train length is not matched.'
    # assert len(txt_test_set) == len(cur_test_set), 'May be test dataset of this benchmark: {} - is removed!'.format(find_dataset_name(dataset_path))
    
    ret = True
    for i in range(2):
        cur_set = cur_sets[i]
        txt_set = txt_sets[i]
        phase = 'train' if 'train' in cur_set[0] else 'test'
        print("Check phase {} in dataset {}".format(phase, find_dataset_name(dataset_path)))
        print(cur_set[0], txt_set[0])
         
        cur_dict = {path: 0 for path in cur_set}
        for path in tqdm(txt_set):
            try:
                if cur_dict[path] == 0:
                    cur_dict[path] = 1
            except:
                cur_dict[path] = -1
        
        cnt_lack = 0
        for path, v in cur_dict.items():
            if v == -1:
                print("{} | in txt-file not exists in current-device!".format(path))
                ret = False
            if v == 0:
                if phase == 'train':
                    # print("{} | in current-device not exists in txt_file!".format(path))
                    print(path)
                    ret = False
                cnt_lack += 1
        print('Lack {} image'.format(cnt_lack))
    return ret
                
def make_dataset_from_txt_file(dataset_path: str, train_file: str, test_file: str, val_file: str, check_sync=True, sync=False):
    txt_train_set = get_image_from_txt_file(train_file, join(dataset_path, 'train'))
    txt_val_set = get_image_from_txt_file(val_file, join(dataset_path, 'train'))
    txt_test_set = get_image_from_txt_file(test_file, join(dataset_path, 'test'))
    
    accepted = True
    if check_sync:
        ret = check_synchronization_between_servers(dataset_path, txt_train_set, txt_val_set, txt_test_set)
        accepted = ret
        print("{} synchronize! \n===================\n".format("accepted" if ret else "denied"))
        
    # Make dataset
    if sync and accepted:
        # Move val
        val_dir = join(dataset_path, 'val')
        if not osp.exists(val_dir):
            os.mkdir(val_dir)
        for img_path in txt_val_set:
            cls = img_path.split('/')[-2]
            cls_dir = join(val_dir, cls)
            if not osp.exists(cls_dir):
                os.mkdir(cls_dir)
            shutil.move(img_path, cls_dir)
            
        # Delete test:
        cur_test_set = glob(join(dataset_path, 'test', '*/*'))
        cur_test_dict = {path: 0 for path in cur_test_set}
        redunt_dir = join(dataset_path, 'redunt_test')
        if not osp.exists(redunt_dir):
            os.mkdir(redunt_dir)
        for path in tqdm(txt_test_set):
            try:
                if cur_test_dict[path] == 0:
                    cur_test_dict[path] = 1
            except:
                cur_test_dict[path] = -1
        
        for path, v in cur_test_dict.items():
            if v == 0:
                shutil.move(path, redunt_dir)
    
def make_validation_set(dataset_path: str, num_real: int, num_fake: int):
    """ Function for some dataset that maintains number of test samples. eg: Celeb-DF, UADFV, df_timit
    Args:
        dataset_path (str): path to dataset
        num_real (int): number of real validation samples that want to make
        num_fake (int): number of fake validation samples that want to make
    """
    train_set = join(dataset_path, 'train')
    val_set = join(dataset_path, 'val')
    if not osp.exists(val_set):
        os.mkdir(val_set)
        
    clses = os.listdir(train_set)
    for cls in clses:
        print(cls)
        val_dir = join(val_set, cls)
        if not osp.exists(val_dir):  
            os.mkdir(val_dir)
        imgs = glob(join(train_set, cls, '*'))
        random.shuffle(imgs)
        num_samples = num_real if 'real' in cls else num_fake
        for _ in tqdm(range(num_samples)):
            img = imgs.pop()
            shutil.move(img, val_dir)
        
    ### CHECK ###
    for cls in os.listdir(val_set):
        print(join(val_set, cls), end=' - ')
        print("Number samples: ", len(os.listdir(join(val_set, cls))))
    
def delete_test_set(dataset_path: str, num_real: int, num_fake: int):
    """ Function to delete some samples for some dataset. eg: dfdc, df_in_the_wild, ff
    Args:
        dataset_path (str): path to dataset
        num_real (int): number of real test samples that want to delete
        num_fake (int): number of fake test samples that want to delete
    """
    test_set = join(dataset_path, 'test')
    for cls in os.listdir(test_set):
        print(cls)
        imgs = glob(join(test_set, cls, '*'))
        random.shuffle(imgs)
        num_samples = num_real if 'real' in cls else num_fake
        for _ in tqdm(range(num_samples)):
            img = imgs.pop()
            os.remove(img)
        
    ### CHECK ###
    for cls in os.listdir(test_set):
        print(join(test_set, cls), end=' - ')
        print("Number samples: ", len(os.listdir(join(test_set, cls))))

def delete_images_from_txt_file(train_file: str, train_dir: str):
    txt_train_set = get_image_from_txt_file(train_file, train_dir)
    if txt_train_set is None:
        print("Not images in file {}".format(train_file))
        return
    imgs = glob(join(train_dir, '*/*'))
    imgs_dict = {img : 0 for img in imgs}
    for image in txt_train_set:
        img_path = join(train_dir, image)
        if not osp.exists(img_path) or img_path not in imgs_dict.keys():
            print("Error image {}".format(image))
            return
        imgs_dict[img_path] = 1

    cnt = 0
    for img, v in imgs_dict.items():
        if v == 1:
            continue
        else:
            cnt += 1
            os.remove(img)
    print("Number of deleted images: ", cnt)

def truncate_images_in_dataset(dataset_path: str, phase='train', real_trunc_ratio=0.0, fake_trunc_ratio=0.0, video_pos=0, id_pos=1, truncated=False):
    dset = join(dataset_path, phase)
    cls = ['1_df', '0_real']
    for c in cls:
        if not osp.exists(join(dset, c)):
            print("Error! Can not find {} dataset.".format(c))
    
    def traverse_names(names, v_pos, id_pos, head_path):
        # Traverse
        res, trunc_accept = {}, True
        for name_ in names:
            name = name_.split('_')
            v, id = "_".join([name[pos] for pos in v_pos]), name[id_pos]
            if v not in res.keys():
                res[v] = [join(head_path, name_)]
            else:
                res[v].append(join(head_path, name_))
        # Check:
        for k, values in res.items():
            if len(values) != len(set(values)):
                print(k, [osp.basename(v) for v in values])
                trunc_accept = False
        return res, trunc_accept

    for c in cls:
        print("Class: ", c)
        cls_dir = join(dset, c)
        trunc_ratio = real_trunc_ratio if 'real' in c else fake_trunc_ratio
        img_names = os.listdir(cls_dir)

        video_frame, trunc = traverse_names(img_names, video_pos, id_pos, cls_dir)
        if not trunc:
            print("Error appears! Can not truncate images.")
            return
        if not truncated:
            print("Permission denied!")
            return
        
        for video, frames in tqdm(video_frame.items()):
            trunc_images = random.sample(frames, k=int(trunc_ratio * len(frames)))
            # print("Images: {}, truncated: {}".format(len(frames), len(trunc_images)))
            for trunc_image in trunc_images:
                os.remove(trunc_image)


def aggregate_fake_ff_set(ff_path: str):
    """ Aggregate all the samples in component of ff dataset
    Args:
        ff_path (str): path to ff dataset
    """
    train_ff = join(ff_path, 'test')
    val_ff = join(ff_path, 'train')
    test_ff = join(ff_path, 'val')
    
    ff_set = [train_ff, val_ff, test_ff]
    for dset in ff_set:
        print("* ", dset)
        fake_agg_dir = join(dset, '1_fake')
        if not osp.exists(fake_agg_dir):
            os.mkdir(fake_agg_dir)
        for fake_type in ['1_df', '1_f2f', '1_fs', '1_nt']:
            fake_type_dir = join(dset, fake_type)
            print('   === ', fake_type)
            for img_name in tqdm(os.listdir(fake_type_dir)):
                img_path = join(fake_type_dir, img_name)
                new_path = join(fake_type_dir, fake_type.replace('1_', '') + '_' + img_name)
                os.rename(img_path, new_path)
                shutil.move(new_path, fake_agg_dir)

def find_dataset_name(dataset_path: str):
    if 'component' in dataset_path:
        return 'ff_' + dataset_path.split('/')[-1]
    if 'ff' in dataset_path:
        return 'ff'
    if 'UADFV' in dataset_path:
        return 'uadfv'
    if 'df_in_the_wild' in dataset_path:
        return 'wild'
    if 'dfdc' in dataset_path:
        return 'dfdc'
    if 'df_timit' in dataset_path:
        return 'timit'
    if 'Celeb-DF' in dataset_path:
        return 'celeb'

def check_equal(file_1: str, file_2: str):
    lst_1 = get_image_from_txt_file(file_1, "")
    lst_2 = get_image_from_txt_file(file_2, "")
    dict_1 = {item_1: 1 for item_1 in lst_1}
    for item_2 in lst_2:
        try:
            if dict_1[item_2] == 1:
                dict_1[item_2] = 2  # item 2 is in list 1
        except:
            dict_1[item_2] = 0      # item 2 is not in list 1
    
    ret = True
    for item, v in dict_1.items():
        if v == 0:
            ret = False
        elif v == 1:
            ret = False
        elif v == 2:
            continue
    return ret

def check_synchronization(statistic_dir: str):
    files = os.listdir(statistic_dir)
    visited = {file: 0 for file in files}
    for file in files:
        if not visited[file]:
            file_1 = join(statistic_dir, file)
            server_1 = file.split('_')[0]
            server_2 = 'server61' if server_1 == '8tcp' else '8tcp'
            file_2 = file_1.replace(server_1, server_2)
            if osp.basename(file_2) not in files:
                print("WRONG FILE!")
                return
            visited[osp.basename(file_1)] = 1
            visited[osp.basename(file_2)] = 1
            ret = check_equal(file_1, file_2)
            if not ret:
                print("File {} and {} has wrong!".format(file_1, file_2))
                break
            else:
                print("File {} and {} are ok!".format(file_1, file_2))

def statistic_video(data_dir: str):
    real_dir = join(data_dir, '0_real')
    fake_dir = join(data_dir, '1_df')
    real_imgs = os.listdir(real_dir)
    fake_imgs = os.listdir(fake_dir)
    real_videos, fake_videos = {}, {}

    for img in real_imgs:
        if '.jpg' not in img:
            print('bug')
            return
        base_name = img.replace('.jpg', '').split('_')
        video_name = '_'.join(base_name[:-1])
        index = base_name[-1]
        if video_name not in real_videos.keys():
            real_videos[video_name] = [join(real_dir, img)]
        else:
            real_videos[video_name].append(join(real_dir, img))

    for img in fake_imgs:
        if '.jpg' not in img:
            print('bug')
            return
        base_name = img.replace('.jpg', '').split('_')
        video_name = '_'.join(base_name[:-1])
        index = base_name[-1]
        if video_name not in fake_videos.keys():
            fake_videos[video_name] = [join(fake_dir, img)]
        else:
            fake_videos[video_name].append(join(fake_dir, img))

    real_len_images = [len(v) for k, v in real_videos.items()]
    fake_len_images = [len(v) for k, v in fake_videos.items()]
    real_mean_len = sum(real_len_images)/len(real_len_images)
    fake_mean_len = sum(fake_len_images)/len(fake_len_images)
    return real_videos, fake_videos, real_len_images, fake_len_images, real_mean_len, fake_mean_len

def move_image_from_train_to_val(dataset_path: str, expected_real_images=5000, expected_fake_images=5000):
    train_dir = join(dataset_path, 'train')
    val_dir = join(dataset_path, 'val')
    for cls in ['0_real', '1_df']:
        cls_train_dir = join(train_dir, cls)
        cls_val_dir = join(val_dir, cls)
        train_img = glob(join(cls_train_dir, '*'))
        num_move = expected_real_images if 'real' in cls else expected_fake_images
        train_img = random.sample(train_img, k=num_move)
        for im in train_img:
            shutil.move(im, cls_val_dir)

def split_by_video(dataset_path: str, val_real_image=25000, val_fake_image=30000, move=False):
    # Merge train and val first:
    train_dir = join(dataset_path, 'train')
    val_dir = join(dataset_path, 'val')
    test_dir = join(dataset_path, 'test')

    train_img_paths = glob(join(train_dir, '*/*'))
    val_img_paths = glob(join(val_dir, '*/*'))
    test_img_paths = glob(join(test_dir, '*/*'))

    for v in val_img_paths:
        cls = v.split('/')[-2]
        shutil.move(v, join(train_dir, cls))
        
    # Statistic videos:
    real_v, fake_v, real_len_images, fake_len_images, real_mean_len, fake_mean_len = statistic_video(train_dir)
    print("******NUmber of real videos in test: ", len(statistic_video(test_dir)[0].keys()))
    print("******NUmber of fake videos in test: ", len(statistic_video(test_dir)[1].keys()))
    print("******Number of all in train real videos: ", len(real_v.keys()))
    # for k, v in real_v.items():
    #     print(k, ' = ', v)
    #     break
    print("******Number of all in train fake videos: ", len(fake_v.keys()))
    # for k, v in fake_v.items():
    #     print(k, ' = ', v)
    #     break
    # print("******Real videos: ", real_len_images)
    print("******Real mean: ", real_mean_len)
    # print("******Fake videos: ", fake_len_images)
    print("******Fake mean: ", fake_mean_len)
    num_val_real_v = int(val_real_image/real_mean_len)
    num_val_fake_v = int(val_fake_image/fake_mean_len)
    print("******Want val real video: ", num_val_real_v)
    print("******Want val fake video: ", num_val_fake_v)
    val_real_video = random.sample(list(real_v.keys()), k=num_val_real_v)
    val_fake_video = random.sample(list(fake_v.keys()), k=num_val_fake_v)
    val_r_v_dict = {k: real_v[k] for k in val_real_video}
    val_f_v_dict = {k: fake_v[k] for k in val_fake_video}
    print("******Expected val real images: ", sum([len(v) for k, v in val_r_v_dict.items()]))
    print("******Expected val fake images: ", sum([len(v) for k, v in val_f_v_dict.items()]))
    # Move:
    if move:
        move_images(val_r_v_dict, dir=join(val_dir, '0_real'))
        move_images(val_f_v_dict, dir=join(val_dir, '1_df'))

def move_images(d: dict, dir: str):
    for v in d.values():
        for img_path in v:
            shutil.move(img_path, dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Filter noise image by another face detection module")
    parser.add_argument("--device_name", type=str, default='server61')
    #
    parser.add_argument("--dataset_path", type=str, required=False, help="path to dataset")
    parser.add_argument("--make_validation_set", type=int, default=0, help="make validation set from train set (maintain test samples) or not")
    parser.add_argument("--num_real_val", type=int, default=0)
    parser.add_argument("--num_fake_val", type=int, default=0)
    #
    parser.add_argument("--delete_test_set", type=int, default=0, help="Delete some samples in test set")
    parser.add_argument("--num_real_test", type=int, default=0)
    parser.add_argument("--num_fake_test", type=int, default=0)
    #
    parser.add_argument("--agg_fake_ff_set", type=int, default=0, help="Aggregate fake samples of ff dataset")
    #
    parser.add_argument("--make_dataset_from_txt_file", type=int, default=0, help="Synchronize among servers")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--val_file", type=str)
    parser.add_argument("--check_sync", type=int, default=1)
    parser.add_argument("--sync", type=int, default=0)
    return parser.parse_args()
    
if __name__ == '__main__':
    # args = parse_args()
    # if args.make_validation_set:
    #     make_validation_set(args.dataset_path, int(args.num_real_val), int(args.num_fake_val))
    # if args.delete_test_set:
    #     delete_test_set(args.dataset_path, int(args.num_real_test), int(args.num_fake_test))
    # if args.agg_fake_ff_set:
    #     aggregate_fake_ff_set(args.dataset_path)
        
    # statisticize_dataset(args.dataset_path)
    # log_dataset_statistic(args.dataset_path, find_dataset_name(args.dataset_path), "/mnt/disk1/phucnp/Graduation_Thesis/review/forensics/preprocess_data/data_statistic", args.device_name)
    # if args.make_dataset_from_txt_file:
    #     make_dataset_from_txt_file(args.dataset_path, args.train_file, args.test_file, args.val_file, check_sync=bool(args.check_sync), sync=bool(args.sync))
    #     statisticize_dataset(args.dataset_path)
    # check_synchronization("/home/phucnp/facial-fake-detection/forensics/preprocess_data/data_statistic")
    
    # WILD
    # print("DF_IN_THE_WILD")
    # dataset_path = "/mnt/disk1/doan/phucnp/Dataset/df_in_the_wildv2/image"
    # statisticize_dataset(dataset_path)
    # # train_dir = "/mnt/disk1/doan/phucnp/Dataset/df_in_the_wildv2/image/train"
    # # phase = "train"
    # # real_trunc_ratio = float(2/3)
    # # fake_trunc_ratio = float(2/3)
    # # video_pos = [0, 1]
    # # id_pos = 2
    # # truncated=True
    # # truncate_images_in_dataset(dataset_path=dataset_path, phase=phase, real_trunc_ratio=real_trunc_ratio, fake_trunc_ratio=fake_trunc_ratio, video_pos=video_pos, id_pos=id_pos, truncated=truncated)
    # # print("After delete df_in_the_wild: ")
    # # statisticize_dataset(dataset_path)
    # device_name = "server61"
    # log_dataset_statistic(dataset_path, "wildv2", "/mnt/disk1/doan/phucnp/Graduation_Thesis/review/forensics/preprocess_data/deleted_statistic", device_name)
    # # print()
    # # print()

    # # DFDC
    # # print("DFDC")
    # dataset_path = "/mnt/disk1/doan/phucnp/Dataset/dfdcv2/image"
    # statisticize_dataset(dataset_path)
    # # train_dir = "/mnt/disk1/doan/phucnp/Dataset/dfdcv2/image/train"
    # # phase = "train"
    # # real_trunc_ratio = float(0)
    # # fake_trunc_ratio = float(2/3)
    # # video_pos = [0]
    # # id_pos = 1
    # # truncated=True
    # # truncate_images_in_dataset(dataset_path=dataset_path, phase=phase, real_trunc_ratio=real_trunc_ratio, fake_trunc_ratio=fake_trunc_ratio, video_pos=video_pos, id_pos=id_pos, truncated=truncated)
    # # print("After delete dfdc: ")
    # # statisticize_dataset(dataset_path)
    # device_name = "server61"
    # log_dataset_statistic(dataset_path, "dfdcv2", "/mnt/disk1/doan/phucnp/Graduation_Thesis/review/forensics/preprocess_data/deleted_statistic", device_name)
    # # print()
    # # print()

    # # Celeb_DF
    # print("Celeb-DF")
    # dataset_path = "/mnt/disk1/doan/phucnp/Dataset/Celeb-DFv2/image"
    # statisticize_dataset(dataset_path)
    # # train_dir = "/mnt/disk1/doan/phucnp/Dataset/Celeb-DFv2/image/train"
    # # phase = "train"
    # # real_trunc_ratio = float(0)
    # # fake_trunc_ratio = float(2/3)
    # # video_pos = [0, 1, 2]
    # # id_pos = 3
    # # truncated=True
    # # truncate_images_in_dataset(dataset_path=dataset_path, phase=phase, real_trunc_ratio=real_trunc_ratio, fake_trunc_ratio=fake_trunc_ratio, video_pos=video_pos, id_pos=id_pos, truncated=truncated)
    # # print("After delete celeb_DF: ")
    # # statisticize_dataset(dataset_path)
    # device_name = "server61"
    # log_dataset_statistic(dataset_path, "celeb-dfv2", "/mnt/disk1/doan/phucnp/Graduation_Thesis/review/forensics/preprocess_data/deleted_statistic", device_name)

    # DFDC:
    dataset_path = "/mnt/disk1/doan/phucnp/Dataset/dfdcv3/image"
    statisticize_dataset(dataset_path)
    expected_real_images = 3000
    expected_fake_images = 3000
    move_image_from_train_to_val(dataset_path=dataset_path, expected_real_images=expected_real_images, expected_fake_images=expected_fake_images)
    # val_real_image = 25000
    # val_fake_image = 28000
    # split_by_video(dataset_path=dataset_path, val_real_image=val_real_image, val_fake_image=val_fake_image, move=True)
    statisticize_dataset(dataset_path)
    # lst = ['1_103_, 2_49_, 4_64_']
    # for img in os.listdir(join(dataset_path, 'train/0_real')):
    #     for l in lst:
    #         if l in img:
    #             print('bug')