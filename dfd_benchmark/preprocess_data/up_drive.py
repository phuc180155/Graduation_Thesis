from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import os.path as osp
import argparse
from tqdm import tqdm

# Train/1_df
targetDirID = "12APtbO_4s275OmCoqvC_l4v2Niwc8DEe"
src = "../../../../2_Deep_Learning/Dataset/facial_forgery/dfdc/image/train/1_df/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_folder", type=int)
    parser.add_argument("--next_folder", type=int)
    return parser.parse_args()

args = parse_args()
pre_folder = args.pre_folder
nxt_folder = args.next_folder

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() 

drive = GoogleDrive(gauth)
exist_file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(targetDirID)}).GetList()

for i in range(pre_folder, nxt_folder): 
    src_dir = src + str(i)
    print("Directory: ", src_dir)
    files = os.listdir(src_dir)
    error = 0
    for file_name in tqdm(files):
        file_path = osp.join(src_dir, file_name)

        # Xoá file nếu không sẽ có 2 file đó
        for file1 in exist_file_list:
            if file1['title'] == file_name:
                file1.Delete()
                print("File {} exists!".format(file_name))
                
        try:
            gfile = drive.CreateFile({'parents': [{'id': targetDirID}], 'title': file_name})
            # Read file and set it as the content of this instance.
            gfile.SetContentFile(file_path)
            gfile.Upload() # Upload the file.
        except:
            error = 1
            print("Load error in subfolder {} from {} to {}".format(i, pre_folder, nxt_folder))
            break
    if error:
        break

    print("Done directory <{}>".format(src_dir))

		
