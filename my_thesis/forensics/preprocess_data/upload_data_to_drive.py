# from pydrive.drive import GoogleDrive
# from pydrive.auth import GoogleAuth
from zdrive import Uploader, Downloader
   
# For using listdir()
import os
   
def upload_with_pydrive():
    # Below code does the authentication
    # part of the code
    gauth = GoogleAuth()
    
    # Creates local webserver and auto
    # handles authentication.
    gauth.LocalWebserverAuth()       
    drive = GoogleDrive(gauth)
    
    # replace the value of this variable
    # with the absolute path of the directory
    path = "E:/test"
    
    # iterating thought all the files/folder
    # of the desired directory
    for x in os.listdir(path):
    
        f = drive.CreateFile({'title': x})
        f.SetContentFile(os.path.join(path, x))
        f.Upload()
    
        # Due to a known bug in pydrive if we 
        # don't empty the variable used to
        # upload the files to Google Drive the
        # file stays open in memory and causes a
        # memory leak, therefore preventing its 
        # deletion
        f = None

def upload_folder_to_drive_by_zdrive(folder_path, dest_id):
    uploader = Uploader()
    result = uploader.uploadFolder(folder_path, max_depth=100, parentId=dest_id)
    print(result)

def download_folder_to_drive_by_zdrive(src_id, dest_dir):
    downloader = Downloader()
    result = downloader.downloadFolder(folderId=src_id, destinationFolder=dest_dir)
    print(result)


if __name__ == '__main__':
    folder_path = "/mnt/disk1/doan/phucnp/Dataset/Celeb-DFv4" #
    dst_id = "1V2V8UPfqy0W5MN4TuR8yMPTEFgMuIRli"    #my repo
    upload_folder_to_drive_by_zdrive(folder_path, dst_id)