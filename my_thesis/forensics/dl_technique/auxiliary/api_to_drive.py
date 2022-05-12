from ast import Return
import os, sys
import os.path as osp
from os.path import join
from click import pass_context

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from scipy.fft import dst
from tqdm import tqdm

import json

from typing import Dict, List
from glob import glob
import shutil

class GoogleDriveAPI(object):
    """
        Class for using some API to manipulate with google drive, like: 
            * upload file/folder to gdrive
            * download file/folder to device
            * delete a file/folder in gdrive
            * find a file/folder in gdrive
            ...
    """
    def __init__(self, root_id: str, method: str, device='61', display_tree=False):
        """
        Args:
            root_id (str): id of root directory 
            method (str): the approach method to connect google drive, in list: ['pydrive']. If method is pydrive, you must use API Console to create a project on 
                        Google Cloud Platform, create credentials with Desktop Application in Application Type and authenticate to your google drive. 
                        The tutorials come from: https://codingshiksha.com/tutorials/python-upload-files-to-google-drive-using-pydrive-library-full-tutorial/
            display_tree (bool): display hierachy of root directory if set to True.
        """
        self.method = method
        self.device = device
        self.drive = None

        # Define drive instance
        if method == 'pydrive':
            gauth = GoogleAuth()
            gauth.CommandLineAuth()
            self.drive = GoogleDrive(gauth)
            
        # Define root folder
        self.root_id = root_id
        try:
            self.root = self.get_file(root_id)
        except:
            self.root = None
        # Traverse all files in root:
        if display_tree and self.root_id != "":
            self.display_hierachical_tree_structure(self.root_id)
        self.paths = self.traverse_folder(self.root_id)
    
    def get_file(self, id: str):
        """Return GoogleDriveFile from an id.
        """
        return self.drive.CreateFile({'id': id})
        
        
    def is_directory(self, id: str):
        """_summary_ Check a file is a directory or not. Returns: Boolean
        """
        file = self.get_file(id)
        return file['mimeType'] == 'application/vnd.google-apps.folder'
    
    def is_file(self, id: str):
        """_summary_ Check a file is a file or not. Returns: Boolean
        """
        file = self.get_file(id)
        return file['mimeType'] != 'application/vnd.google-apps.folder'
    
    def listdir(self, folder_id: str):
        """_summary_: return all files in a folder having id <folder_id>
        Returns:
            file_list(
                List[
                    GoogleDriveFile({
                        'id': '<file_id>',
                        'alternateLink': <full link of this file/folder>,
                        'title': <name of file/folder>,
                        'parents': <parent folder of this file/folder> [
                            'id': '<parent_id>',
                            ...
                        ],
                        'ownerNames': List[str], like: ['Phúc Nguyễn Phi', ...]
                    })
                ]
            )
        """
        assert self.is_directory(folder_id), "This item is not a folder."
        return self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(folder_id)}).GetList()

    def create_file(self, file_name: str, content: str, dst_id: str, overwrite=False):
        """_summary_ Create a file with content in destination directory. Return: id of created file
        """
        assert self.is_directory(dst_id), "Error! Destination should be a folder"

        exist = True if len(self.find_file_in_folder(file_name, dst_id)) else False
        if overwrite:
            file_list = self.listdir(folder_id=dst_id)
            try:
                for file in tqdm(file_list):
                    if file['title'] == file_name:
                        print("Found existing files in gdrive.")
                        file.Trash()
            except:
                pass
        if not overwrite and exist:
            file_name = self.set_copied_file_name(file_name, dst_id)

        file = self.drive.CreateFile(metatdata={
            'parents': [{
                'id': dst_id
            }],
            'title': file_name
        })
        file.SetContentString(content)
        file.Upload()
        return file['id']
        
    def create_folder(self, folder_name: str, dst_id: str):
        """_summary_ Create a folder in destination directory. Returns: id of created folder.
        """
        assert self.is_directory(dst_id), "Error! Destination should be a folder"
        folder = self.drive.CreateFile(metadata={
            'parents': [{
                'id': dst_id
            }],
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        })
        folder.Upload()
        return folder['id']
    
    def delete_file(self, id: str, permanent=True):
        """_summary_ Delete a file/folder
        """
        file = self.get_file(id=id)
        if not permanent:
            file.Trash()
        else:
            file.Delete()
    
    def delete_file_in_folder(self, file_name: str, folder_id: str, permanent=False, verbose=True):
        """_summary_ Delete a file/folder with name <file_name> in folder has id <folder_id>
        """
        assert self.is_directory(folder_id), "The second parameter must be a folder"
        files = self.listdir(folder_id)
        for file in files:
            if file['title'] == file_name:
                if verbose:
                    print('File {} found! Id = {}'.format(file['title'], file['id']))
                if not permanent:
                    file.Trash()
                else:
                    file.Delete()
            
    def upload_file_to_drive(self, file_path: str, dst_id="", overwrite=True):
        """Upload a file from this device to drive. If not have destination folder, default sets root directory. Return: id of created file.
        """
        # print("Uploading file {}...".format(file_path))
        dst_id = self.root_id if dst_id == "" else dst_id
        assert self.is_directory(dst_id), "Destination should be a folder."
        assert osp.exists(file_path), "Source not exist."
        
        # Delete existing file before overwrite:
        if overwrite:
            self.delete_file_in_folder(file_name=osp.basename(file_path), folder_id=dst_id, verbose=True)

        # Upload file to drive:
        file_name = osp.basename(file_path)
        exist = True if len(self.find_file_in_folder(file_name, dst_id)) else False
        if not overwrite and exist:
            file_name = self.name_copied_file(file_name)

        f = self.drive.CreateFile({
            'title': file_name,
            'parents': [{
                'id': dst_id
            }]
        })
        f.SetContentFile(filename=file_path)
        f.Upload()
        return f['id']
    
    def upload_folder_to_drive(self, folder_path: str, dest_id: str, exclude_folder=None, overwrite=True, merge=False):
        """_summary_ upload a folder to gdrive
        Args:
            folder_path (str): path to folder
            dst_id (str): id of folder in gdrive
            overwrite (bool, optional): If overwrite, folder in drive will be replace base on "merge" parameter:
                If merge=True, 2 folder will be merge, 2 files have the same name will be added 'copy' in one of their name. 
                If merge=False, uploaded folder will completely replace current folder in gdrive. 
                If not overwrite, another with the name is concatenated with 'copy' will be created.
                Defaults to False.
        """
        print("Uploading folder {}...".format(folder_path))
        assert self.is_directory(id=dest_id), "Error! Destination should be a folder."
        assert osp.isdir(folder_path), "Error! Source should be a folder."
            
        def upload(src_folder: str, dst_id: str, root=False):
            """_summary_ Upload all files in a src folder to destionation dir.
                Args:
                    "root=False" determines src_folder is the root folder of uploaded folder (folder_path).
                    Otherwise, "root=True" determines src_folder is a sub-folder of uploaded folder
            """
            # print("Upload ", src_folder, "to", dst_id)
            # Upload folder first:
            if exclude_folder is not None and isinstance(exclude_folder, list):
                if src_folder in exclude_folder:
                    return ""

            folder_name = osp.basename(src_folder)
            folders = self.find_file_in_folder(folder_name, dst_id)
            assert len(folders) < 2, "Must be less than 2 folders with the same name in one destinaton directory."
            exist = True if len(folders) else False
            folder_id = "" if not exist else folders[0]['id']
            create_file = False
            if not exist:
                create_file = True
                
            if root:
                if exist:
                    if not overwrite:
                        folder_name = self.set_copied_file_name(file_name=folder_name, dest_id=dst_id)
                        create_file = True
                    else:
                        if not merge:
                            self.delete_file(id=folder_id, permanent=False)
                            create_file = True
            else:
                if exist:
                    if overwrite and merge:
                        folder_name = self.set_copied_file_name(file_name=folder_name, dest_id=dst_id)
                        create_file = True
            # Create file if has flag
            if create_file:
                folder_id = self.create_folder(folder_name, dst_id)
                # print("Create folder: ", folder_name)              
            src_files = os.listdir(src_folder)
            for src_file in tqdm(src_files):
                src_path = join(src_folder, src_file)
                if osp.isfile(src_path):
                    self.upload_file_to_drive(file_path=src_path, dst_id=folder_id, overwrite=False)
                else:
                    upload(src_path, folder_id, root=False)
            return folder_id
        f_id = upload(src_folder=folder_path, dst_id=dest_id, root=True)
        return f_id

    def download_file_to_device(self, file_path: str, dest_dir: str, overwrite=False):
        """_summary_ download a file with path <file_path> to destination directory.
        """
        print("Downloading...")
        assert osp.isdir(dest_dir), "Destination should be a folder."
        files = self.find_file(file_path, verbose=True)
        assert len(files) == 1, "{} files found!".format("No" if len(files) == 0 else len(files))
        assert self.is_file(files[0]['id']), "Source should be a file."
        
        # Check destination contains file has the same name or not.
        file_name = osp.basename(file_path)
        dest_path = join(dest_dir, file_name)
        exist = osp.exists(dest_path)
        if overwrite and exist:
            os.remove(dest_path)
            
        # Redefine the name if file existed and overwrite is set to False.
        if not overwrite and exist:
            file_name = self.name_copied_file(file_name)
            while osp.exists(join(dest_dir, file_name)):
                file_name = self.name_copied_file(file_name)  

        # Download file
        file = self.get_file(id=files[0]['id'])
        file.GetContentFile(filename=file_name)
        shutil.move(src=file_name, dst=join(dest_dir, file_name))
    
    def download_folder_to_device(self, folder_path: str, dest_dir: str, overwrite=False):
        """_summary_ Download a folder from gdrive to destination directory. 
            If overwrite=True and the folder has existed in destination dir, the folder will be overwritten completely. Otherwise, another folder with suffix ' - Copy' in name will be created.
        """
        assert osp.isdir(dest_dir), "Destination should be a folder!"
        folders = self.find_file(folder_path, verbose=True)
        assert len(folders) == 1, "Error!{} folders found!".format(len(folders) if len(folders) else "No")
        assert self.is_directory(folders[0]['id']), "Source should be a folder!"
        
        def download(src_dir: str, dst_dir: str, root=False):
            # Create folder with name <src_dir.name> in dst_dir:
            dst_path = ""
            if root:
                # Check if destination dir contains downloaded folder
                src_name = osp.basename(src_dir)
                dst_path = join(dst_dir, src_name)
                exist = osp.exists(dst_path)
                if exist and overwrite:
                    shutil.rmtree(dst_path)
                    
                # Redefine the name if folder existed and overwrite is set to False.
                if not overwrite and exist:
                    src_name = self.name_copied_file(src_name)
                    while osp.exists(join(dst_dir, src_name)):
                        src_name = self.name_copied_file(src_name) 
                dst_path = join(dst_dir, src_name)
            else:
                dst_path = join(dst_dir, osp.basename(src_dir))
            os.mkdir(dst_path)
            # Copy files in src_dir to new created folder.
            src_files = self.listdir(self.find_file(src_dir, verbose=True)[0]['id'])
            for src_file in src_files:
                if self.is_file(src_file['id']):
                    src_file.GetContentFile(filename=src_file['title'])
                    shutil.move(src=src_file['title'], dst=join(dst_path, src_file['title']))
                else:
                    download(src_dir=join(src_dir, src_file['title']), dst_dir=dst_path, root=False)
                    
        download(src_dir=folder_path, dst_dir=dest_dir, root=True)
            
    def traverse_folder(self, folder_id: str):
        """_summary_ Traverse a file/folder and returns a dictionary with KEY is file path and VALUE is a List contains those files'id.
        """
    
        def traverse(cur_id: str, cur_folder_name: str, traverse_dict: Dict[str, List[str]]):
            traverse_dict[cur_id] = cur_folder_name    
            if self.is_directory(cur_id):
                list_file = self.listdir(folder_id=cur_id)
                for file in list_file:
                    file_name = cur_folder_name + '/' + file['title']
                    traverse(file['id'], file_name, traverse_dict)
                    
        result = {}
        try:
            folder = self.get_file(folder_id)
            traverse(folder['id'], folder['title'], result)
        except:
            return result
        return result
           
    def find_file(self, file_path: str, verbose=False):
        """_summary_ Find a file with specific path on hierachical structure with root has id <folder_id>
            Returns: List[str]: list of file_ids
        """
        result = []
        for id, path in self.paths.items():
            if (file_path in path) and (osp.basename(file_path) == osp.basename(path)) and all(sub in path.split('/') for sub in file_path.split('/')):
                result.append({
                    'id': id,
                    'path': path
                })
                
        if verbose:
            if len(result) == 0:
                print("File not found!")
            else:
                print("{} {} found: ".format(len(result), "file" if len(result) == 1 else "files"))
                for r in result:
                    print("   {} with id={}".format(r['path'], r['id']))
        # print(result)
        return result
        
    def find_file_in_folder(self, file_name: str, folder_id: str, verbose=False):
        """_summary_ Find all files with name <file_name> in a folder
        """
        files = self.drive.ListFile({'q': "'{}' in parents and title='{}' and trashed=false".format(folder_id, file_name)}).GetList()
        if verbose:
            if len(files) == 0:
                print("File not found!")
            else:
                print("{} {} found. ".format(len(files), "file" if len(files) == 1 else "files"))  
        return files
        
    def display_hierachical_tree_structure(self, folder_id: str, save=True, indent=4):
        """_summary_ Display structure of a folder. 
        """
        def display(id: str, n_tab: int):
            tabs = "".join([" "] * indent * n_tab)
            file = self.get_file(id)
            s = tabs + file['title'] + '\n'
            print(s)
            if self.is_directory(id):
                list_file = self.listdir(folder_id=id)
                for file in list_file:
                    s += display(file['id'], n_tab + 1)
            return s
        print("\n**************** Hierachy structure: ****************")
        hierachy = display(id=folder_id, n_tab=0)
        if save:
            with open("result/hierachy_'{}'.txt".format(folder_id), 'w') as f:
                f.write(hierachy)
        print("*****************************************************")
        
    def name_copied_file(self, file_name: str):
        """_summary_ Rename for a copied file. Returns: new file name
        """
        head, tail = osp.split(file_name)
        name, ext = osp.splitext(tail)
        return join(head, name + ' - ' + self.device + ext)
    
    def set_copied_file_name(self, file_name: str, dest_id: str):
        """_summary_ Rename for a copied file with condition this name doesn't present in destionation dir. Returns: new file name
        """
        while len(self.find_file_in_folder(file_name, dest_id, verbose=False)):
            file_name = self.name_copied_file(file_name)
        return file_name
            
    def check_synchronization(self, gdrive_id: str, device_path: str, exclude_gdrive: List[str], exclude_device: List[str]):
        """_summary_ Check synchronization between two files/folders in google drive and this device. Returns bool
        """
        print("Check synchronization...")
              
        def traverse_in_device(path: str):
            res = [path]
            for fname in os.listdir(path):
                f_path = join(path, fname)
                if osp.isfile(f_path):
                    res.append(f_path)
                else:
                    res.extend(traverse_in_device(f_path))
            return res
            
        def remove_exclude_path(overall: List[str], exclude: List[str]):
            if exclude is None:
                return overall
            # print("overall: ", overall)
            # print("exclude: ", exclude)
            res = []
            for path in overall:
                append = True
                for ex in exclude:
                    if ex in path:
                        append = False
                        break
                if append:
                    res.append(path)
            return res
        
        def check_identical(list_1: List[str], list_2: List[str], save=False):
            print("list 1 - drive: {}, list 2 - device: {}".format(len(list_1), len(list_2)))
            print("* In drive but not in device: ")
            ret = True
            f = open('result/synchronization.txt', 'a') if save else None
            f_drive = open('result/in_drive.txt', 'a') if save else None
            f_device = open('result/in_device.txt', 'a') if save else None
            if save:
                f.write("################ In drive but not in device: \n")
            for l in list_1:
                if l not in list_2:
                    print(l)
                    ret = False
                    if save:
                        f.write(l + '\n')
                if save:
                    f_drive.write(l + '\n')

            if save:
                f.write("\n\n################ In device but not in gdrive: \n")

            print("In device but not in gdrive: ")  
            for l in list_2:
                if l not in list_1:
                    print(l)
                    ret = False
                    if save:
                        f.write(l + '\n')
                if save:
                    f_device.write(l + '\n')
            if save:
                f.close()
                f_drive.close()
                f_device.close()
            return ret
    
        # Drive
        hierachy_drive = self.traverse_folder(folder_id=gdrive_id)
        hierachy_drive_ = [v for v in hierachy_drive.values()]
        hierachy_drive_ = remove_exclude_path(overall=hierachy_drive_, exclude=exclude_gdrive)
        # Device
        hierachy_device = remove_exclude_path(overall=traverse_in_device(path=device_path), exclude=exclude_device)
        head = '/'.join(device_path.split('/')[:-1]) + '/'
        hierachy_device_ = [p.replace(head, '') for p in hierachy_device]    

        final_drive_ = sorted(hierachy_device_)
        final_device_ = sorted(hierachy_drive_)
        return check_identical(final_drive_, final_device_, save=True)

        
if __name__ == '__main__':
    root_id = ""
    print("/".join(["*"]*0))
    print(join("1", ""))
    # root_id = "17Y6jdy74CQEpH2W-fDBHbcv5Duz-cAgT"
    gdrive = GoogleDriveAPI(root_id=root_id, method='pydrive', device='61', display_tree=False)
    
    ################################ TEST Upload Folder ################################
    file_path = "/mnt/disk1/phucnp/Graduation_Thesis/review/forensics/dl_technique/test/test.txt"
    dst_id = "1UKZa6PFKa8uqn0HsfXWE_8f7zmq4nAQd"
    
    # folder_path = "/mnt/disk1/phucnp/Graduation_Thesis/review/forensics/dl_technique/checkpoint" #
    # dst_id = "1XYvccxaHguOUFJ3JLGaIvxisMo0JYAYY"    #my repo
    # merge_id = gdrive.upload_folder_to_drive(folder_path=folder_path, dest_id=dst_id, overwrite=True, merge=True)
    
    # folder_path = "/mnt/disk1/phucnp/Graduation_Thesis/review/forensics/dl_technique/checkpoint" #
    # dst_id = "1QzAdW_4VFMeiLGoFowgb1lNQ4iFSazWA"    #my repo
    # exclude = None
    # merge_id = gdrive.upload_folder_to_drive(folder_path=folder_path, dest_id=dst_id, exclude_folder=exclude, overwrite=False, merge=True)

    # gdrive.display_hierachical_tree_structure(root_id)
    #####################################################################################

    ################################ TEST Download Folder ################################
    # file_path = "dual_efficient_vit/(0.81-12)_v_cross_attention-freq-add_w_0.8_lr_0.0003_patch_2_h_3_d_4_es_none_loss_bce_freeze_0/epoch/best_test_acc_0.806301.pt"
    # dest_path = "/mnt/disk1/phucnp/Graduation_Thesis/review/forensics/dl_technique/test/api_to_drive"
    # gdrive.download_file_to_device(file_path=file_path, dest_dir=dest_path, overwrite=False)
    
    # folder_path = "dual_efficient_vit/(0.81-12)_v_cross_attention-freq-add_w_0.8_lr_0.0003_patch_2_h_3_d_4_es_none_loss_bce_freeze_0" #
    # dest_path = "/mnt/disk1/phucnp/Graduation_Thesis/review/forensics/dl_technique/test/api_to_drive"
    # gdrive.download_folder_to_device(folder_path=folder_path, dest_dir=dest_path, overwrite=False)
    
    # file_path = "my repo/server_61/checkpoint/df_in_the_wild/dual_efficient_vit/(0.81-12)_v_cross_attention-freq-add_w_0.8_lr_0.0003_patch_2_h_3_d_4_es_none_loss_bce_freeze_0/step/best_test_avg_f1_0.801762.pt"
    # dest_path = "/mnt/disk1/phucnp/Graduation_Thesis/review/forensics/dl_technique/test/api_to_drive"
    # gdrive.download_file_to_device(file_path=file_path, dest_dir=dest_path, overwrite=False)

    # gdrive.display_hierachical_tree_structure(root_id)
    #####################################################################################
    
    ################################ Check synchronization ################################
    
    gdrive_id = "1ijcDB7j_2_-U5Px8X9fInQVKW_pdvAJ-"
    device_path = "/mnt/disk1/phucnp/Graduation_Thesis/review/forensics/dl_technique/checkpoint/df_in_the_wild/dual_efficient_vit/(0.81-12)_v_cross_attention-freq-add_w_0.8_lr_0.0003_patch_2_h_3_d_4_es_none_loss_bce_freeze_0"
    sync = gdrive.check_synchronization(gdrive_id=gdrive_id, device_path=device_path)
    print("Synchronize: ", "{}".format("Yes " if sync else "No"))
    #####################################################################################