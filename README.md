
# Camera and LiDAR Calibration use Transformer Architecture
This is a Master Thesis project that tries to calibrate the Camera and LiDAR without using prior knowledge about intrinsic and extrinsic parameters. It will implement ideas from different papers such as Vision Transformer, Stratify, PointNet++,... to understand spatial structure from point clouds and images.

## Libraries Installation
```bash
pip3 install --no-cache-dir -r requirements.txt
pip3 install --no-cache-dir --no-deps torch-points-kernels torch-points3d
pip3 install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## Docker
Pull the Docker image of this repository.

Develop Docker Image:
```bash
docker pull tuyethan/camlidtransformer:cuda118-cudnn8-python310-ubu22.04-deve
```
Runtime Docker Image:
```bash
docker pull tuyethan/camlidtransformer:cuda118-cudnn8-python310-ubu22.04-runtime
```
Or build using Docker File and requirements.txt:
 - Docker/Dockerfile
 - Docker/requirements.txt
   
## Deployment
To start training with one GPU:
```bash
  python3 CamLidCalib_Trans/check_data_dir.py
```
To start training with multi GPU, one node:
```bash
    accelerate launch --config_file CamLidCalib_Trans/config/multiGPU_accelerate.yaml CamLidCalib_Trans/train.py
```
To start training with multi-GPU, use SLURM:
```bash
    sbatch CamLidCalib_Trans/RUN.sbat
```
You can control training parameters via :
 - config/ TransCalib_parameter.yaml

## Dataset
 - Use DataSetBuilders folder to download Dataset.
 - Use check_data_dir.py to check whether it has a suitable directory.
 - Set data_root inside TransCalib_parameter.yaml to your_data_directory

## Result, Logs and Checkpoint
 - You can set logs and PCD results inside TransCalib_parameter.yaml (prj_dir), the default directory will be CamLidCalib_Trans/result.
 - If use multi_gpu_tr and logging_type, the training parameter will also automatically save inside prj_dir/save_writter_path.
 - The checkpoints will be saved in prj_dir/checkpoints, you can resume from checkpoints (resume_from_checkpoint:1,2,...) or start training from scratch (resume_from_checkpoint: null)

