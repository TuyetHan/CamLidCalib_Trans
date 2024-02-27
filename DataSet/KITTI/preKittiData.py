import glob
import os
import random

from .KittiData import KITTIData


class PreKittiData():
    def __init__(self, root_dir:str = None):
        assert root_dir != None, "You need to specify the dataset path"
        self.root_dir = root_dir

    def getSamples(self, valide = 'test'):
        dPath = os.path.join(self.root_dir, valide)
        files = []
        for data_folder in os.listdir(dPath):
            data_folderfp = os.path.join(dPath, data_folder)

            sync_folders = glob.glob(os.path.join(data_folderfp, '*_sync'))
            sync_folders.sort()
            calib_files_dict = {os.path.basename(calib_file):calib_file for calib_file in glob.glob(os.path.join(data_folderfp, '*.txt'))}

            for sync_folder in sync_folders:
                pathImg = os.path.join(sync_folder, 'image_02/data')
                pathPointCloud = os.path.join(sync_folder, 'velodyne_points/data')

                pts_fnames = glob.glob(os.path.join(pathPointCloud, '*.bin'))
                pts_fnames.sort()
                img_fnames = glob.glob(os.path.join(pathImg, '*.png'))
                img_fnames.sort()

                for pts, img in zip(pts_fnames, img_fnames):
                    sample = {
                        'PointCloud_path':pts, 
                        'Image_path':img,
                        'calib_cam_to_cam':calib_files_dict['calib_cam_to_cam.txt'],
                        'calib_velo_to_cam':calib_files_dict['calib_velo_to_cam.txt']
                        }
                    files.append(sample)
        if not files:
            print(f'No files in {dPath}')
            exit(-1)
        return files
    

    def getData(self, valid:bool=False):
        files = self.getSamples('test' if valid else 'train')
        if not valid:
            files *= 4
            random.shuffle(files)
            
        return KITTIData(files)