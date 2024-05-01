import os
import glob
import yaml
from config import config
# from config.utils import *

def debug_check(dPath):
    print(f"Current working directory: {os.getcwd()}")

    # Check if dPath exists
    if not os.path.exists(dPath):
        print(f"Error: {dPath} does not exist.")
        return
    
    # Check if the current process has access to dPath
    if not os.access(dPath, os.R_OK):
        print(f"Error: The current process does not have read access to {dPath}.")
        return
    
    # Check for train and test subdirectories in dPath
    train_dir = os.path.join(dPath, 'train')
    test_dir = os.path.join(dPath, 'test')
    if not os.path.exists(train_dir) and not os.path.exists(test_dir):
        print(f"Error: {train_dir} and {test_dir} does not exist.")
        return
    train_test_dir = train_dir if os.path.exists(train_dir) else test_dir

    # Print all files and directories in dPath
    if os.listdir(train_test_dir) == []:
        print(f"Error: No files or directories found in {train_test_dir}. Should have folders name '2011-xx-xx")
        return
    

    for data_folder in os.listdir(train_test_dir):
        # Check sync folders in data folder
        data_folderfp = os.path.join(train_test_dir, data_folder) 
        print("Check data folder: ",data_folderfp)

        # Check for .txt files in data_folder
        print("Check .txt files in data folder.")
        txt_files = [f for f in os.listdir(data_folderfp) if f.endswith('.txt')]
        if not txt_files:
            print(f"Error: No .txt files found in {data_folderfp}")
            return
        else:
            print("Seem good. List txt_files: ",txt_files)



        sync_folders = glob.glob(os.path.join(data_folderfp, '*_sync'))
        if sync_folders == []:
            print(f"Error: No []_sync directories found in {data_folderfp}")
            return
        else:
            print("[]_sync folders are available. Good")

        for sync_folder in sync_folders:
            pathImg = os.path.join(sync_folder, 'image_02/data')
            if not os.path.exists(pathImg):
                print(f"Error: {pathImg} does not exist.")
                return
            else:
                print("Image folder is available. Good")

            pathPointCloud = os.path.join(sync_folder, 'velodyne_points/data')
            if not os.path.exists(pathPointCloud):
                print(f"Error: {pathPointCloud} does not exist.")
                return
            else:
                print("Point cloud folder is available. Good")


            pts_fnames = glob.glob(os.path.join(pathPointCloud, '*.bin'))
            img_fnames = glob.glob(os.path.join(pathImg, '*.png'))
            if pts_fnames == []:
                print(f"Error: No .bin files found in {pathPointCloud}")
                return
            if img_fnames == []:
                print(f"Error: No .png files found in {pathImg}")
                return
    print("All checks passed. Your dataroot is ready for use.")

# config_file = 'MyRepo/CamLidCalib_Trans/config/TransCalib_parameter.yaml'
config_file = 'CamLidCalib_Trans/config/TransCalib_parameter.yaml'

args = config.get_parser(config_file)
debug_check(args.data_root)