from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import os.path as osp
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--targetDirID", type=str)
    return parser.parse_args()

args = parse_args()
src = args.src
targetDirID = args.targetDirID


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() 

drive = GoogleDrive(gauth)
# exist_file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(targetDirID)}).GetList()

src_dir = src
print("Directory: ", src_dir)
files = os.listdir(src_dir)
error = 0
for file_name in tqdm(files):
    file_path = osp.join(src_dir, file_name)

    # Xoá file nếu không sẽ có 2 file đó
    # for file1 in exist_file_list:
    #     if file1['title'] == file_name:
    #         file1.Delete()
    #         print("File {} exists!".format(file_name))
    #         continue
            
    try:
        gfile = drive.CreateFile({'parents': [{'id': targetDirID}], 'title': file_name})
        # Read file and set it as the content of this instance.
        gfile.SetContentFile(file_path)
        gfile.Upload() # Upload the file.
    except:
        error = 1
        print("Load error in subfolder {} from {} to {}".format(i, pre_folder, nxt_folder))
        break

print("Done directory <{}>".format(src_dir))

		
