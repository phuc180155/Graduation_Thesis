from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import os.path as osp
import argparse
from tqdm import tqdm

###################################### DFDC DATASET #########################################
############## Val/0_real
# targetDirID = "1mNtbeV3pLM2_8WGs7SfbpI391ClAC4b0"
# num_folder = 0->69
# src = "../../../../2_Deep_Learning/Dataset/facial_forgery/dfdc/image/val/0_real/"
############## Val/1_df
# targetDirID = "1M2hqmGTUJ4hcDMuFNvF8F6DHEveCG_kR"
# num_folder = 0->209
# src = "../../../../2_Deep_Learning/Dataset/facial_forgery/dfdc/image/val/1_df/"

############## Train/0_real
# targetDirID = "1TnkR3MXkfOAv9iewobXISZPCmTsT46xR"
# num_folder = 0->1192
# src = "../../../../2_Deep_Learning/Dataset/facial_forgery/dfdc/image/train/0_real/"

############## Train/1_df
# targetDirID = "1QKkDbKDUiwA0-RsIJl7vdD5qp_6airAP"
# num_folder = 0->1192
# src = "../../../../2_Deep_Learning/Dataset/facial_forgery/dfdc/image/train/1_df/"

############## Test/0_real
# targetDirID = "1UcBHl-0ILm3rKr8d8kZOgRhptUK90zgG"
# num_folder = 0->77
# src = "../../../../2_Deep_Learning/Dataset/facial_forgery/dfdc/image/test/0_real/"
############## Test/1_df
# targetDirID = "1qB8LTrmd4oCVwMObjfNlnDORtS45VRP-"
# num_folder = 0->363
# src = "../../../../2_Deep_Learning/Dataset/facial_forgery/dfdc/image/test/1_df/"
###############################################################################################


###################################### DF_IN_THE_WILD DATASET ######################################### (Chưa load xong - sử dụng tạm cho server ngrok)
############## Val/0_real
# targetDirID = "1jjXaku-_qdGnGDs0s9pi4s_DthOBvmkG"
# src = "/home/tampm/df_in_the_wild/image_jpg/val/0_real"
############## Val/1_df
# targetDirID = "1UNqgfMf0NKFPUSK19mP1ZwFaCYyjOw6P"
# src = "/home/tampm/df_in_the_wild/image_jpg/val/1_df"

############## Train/0_real
# targetDirID = "1wcQORCaYXeqSjGQbauhfU-UXPLg7ExB1"
# src = "/home/tampm/df_in_the_wild/image_jpg/train/0_real"
############## Train/1_df
# targetDirID = "1ePORmVG6zNG4gwZwx3ZXAOTGHKR1Dtne"
# src = "/home/tampm/df_in_the_wild/image_jpg/train/1_df"

############## Test/0_real
# targetDirID = "1unNr_3CNb9uhIV-PvLaMXAvqvYf19T1w"
# src = "/home/tampm/df_in_the_wild/image_jpg/test/0_real"
############## Test/1_df
# targetDirID = "1nyU4jnbY8x9skuwO7wgHWvdftJxv-usY"
# src = "/home/tampm/df_in_the_wild/image_jpg/test/1_df"
###############################################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--targetDirID", type=str)
    parser.add_argument("--pre_folder", type=int)
    parser.add_argument("--next_folder", type=int)
    return parser.parse_args()

args = parse_args()
src = args.src
targetDirID = args.targetDirID

pre_folder = args.pre_folder
nxt_folder = args.next_folder

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() 

drive = GoogleDrive(gauth)
# exist_file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(targetDirID)}).GetList()

for i in range(pre_folder, nxt_folder): 
    src_dir = src + str(i)
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
    if error:
        break

    print("Done directory <{}>".format(src_dir))

		
