# CamLidCalib_Trans
Motivation: 
The task of combining between camera and lidar to create a comprehensive 3D perception of the environment is crucial for autonomous driving. Currently, most systems are calibrated manually by technicians which consumes a lot of effort and time, Successfully addressing this challenge will unlock the possibility of autonomous driving.

Target:  
•	Improving the synchronization between lidar and camera in real-time scenarios; therefore, providing a reliable environment representation.
•	The result might be used for generalizing calibration across different types of lidar, cameras. Current calibration requires prior knowledge intrinsic parameters of camera, which prevent automation of calibration process. The network is expected to predict the intrinsic as well as extrinsic parameters by itself.
•	The result might serve as input for more complex calibration problems and 3D environments mapping such as merging point clouds and images between several lidars and cameras.

Component Explain:
DataSetBuilders: Download and Extract Dataset
DataSet: Contains .py for preprocessing data
Models: Contains all model and submodels. Current main model is TransformerCalib
config: Contains files for read .YAML,... task related to configuration.

Current required library:
 - numpy, torch
 - pip install torchnet pytorch_metric_learning
 - pip install wheel
 - pip install torch-scatter torch-cluster torch-sparse torch_geometric torchmetrics
 - pip install omegaconf wandb plyfile hydra-core h5py pandas gdown
 - pip install --no-deps torch-points-kernels
 - pip install --no-deps torch-points3d
 - pip install accelerate
 - pip install matplotlib numba open3d tensorboard scikit-image
