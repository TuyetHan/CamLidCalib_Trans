import os
import functools
import argparse
from config import config
from config.utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataSet.KITTI.preKittiData import PreKittiData

from Models.Encoder import EncoderBlock
from Models.TransformerCalib import TransformerCalib
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import ProjectConfiguration
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch, 
    MixedPrecision, 
    ShardingStrategy, 
    StateDictType,
    FullOptimStateDictConfig, 
    FullStateDictConfig,
    ShardedStateDictConfig,
    ShardedOptimStateDictConfig,
)
config_file = 'CamLidCalib_Trans/config/Trans_Calib_1st.yaml'

def train(model=None, train_loader:DataLoader=None, device:torch.device='cuda', 
          save:bool=True, writer:SummaryWriter=None, epochs:int=100, optimizer:torch.optim.Adam=None, 
          scheduler:torch.optim.lr_scheduler.StepLR=None, minLoss=torch.inf, args:argparse.Namespace=None):
    
    # Initialize constant data
    total_data_loaded = len(train_loader)
    print_freq = args.print_log_freq
    All_batch_size = args.batch_size

    correspondance_point_loss = nn.MSELoss(reduction='none')
    num_recursive_iter = args.num_recursive_iter
    epochs = args.epochs

    model.train()
    if args.multi_gpu_tr == True:
        if (args.resume_from_checkpoint is not None):
            accelerator.load_state()
            accelerator.project_configuration.iteration = args.resume_from_checkpoint
        else:
            # Save the init state
            accelerator.save_state()

    if args.logging_type is not None:
        hyper_paras = config.load_train_parameter(config_file)
        accelerator.init_trackers("Transformer_Calib", config=hyper_paras)
    
    for epoch in range(0, epochs):
        if args.multi_gpu_tr == True:
            print('Node', accelerator.process_index , f'Training epoch {epoch + 1}')
        else:
            print(f'Training epoch {epoch + 1}')

        running_loss = 0.0
        genData = None
        epochLosses = 0.0
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            #todo: We could avoid this line since we set the accelerator with `device_placement=True`.
            img, depth, feat = data['Image'].to(device), data['Depth'].to(device), data['Feature'].to(device)
            transform, intrinsics = data['Transform'].to(device), data['intrinsics'].to(device)

            resT = torch.eye(4,4, device=device, requires_grad=True).repeat(All_batch_size, 1, 1)

            loss = 0
            for _ in range(num_recursive_iter):
                new_depth = depth[:,:3,:].permute(0,2,1)
                out = model(img, new_depth, feat)

                out = genTransformMat(out)
                resT = torch.bmm(resT, out)
                genData = gen_point_cloud_img(depth, resT, transform, intrinsics, img.shape)
                loss_pointcloud_mse = torch.sqrt(correspondance_point_loss(genData['pcd']['pred'], genData['pcd']['exp']).sum(-1)).sum(-1).mean()

                loss += (loss_pointcloud_mse)

            if args.multi_gpu_tr == True:
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()

            running_loss += (loss.item() / num_recursive_iter)
            epochLosses += (loss.item() / num_recursive_iter)
            if i % print_freq == 0:    
                pred_tran = resT[:, :3, 3].detach().cpu()
                exp_trans = -transform[:, :3, 3].detach().cpu()
                pred_rot = resT[:, :3, :3].detach().cpu()
                exp_rot = transform[:, :3, :3].permute(0, 2, 1).detach().cpu()

                running_loss /= print_freq
                geodesic_dist = rotationLoss(pred_rot, exp_rot)
                X_diff, Y_diff, Z_diff = translaitionLoss(pred_tran, exp_trans)
                Yaw_diff, Pitch_diff, Roll_diff = anglesLoss(pred_rot, exp_rot)

                if args.logging_type is not None:
                    accelerator.log({"train_loss": running_loss, 
                                    "train_geodesic_loss": geodesic_dist,
                                    "train_X_loss":X_diff, 
                                    "train_Y_loss":Y_diff, 
                                    "train_Z_loss":Z_diff,
                                    "train_Yaw_loss":Yaw_diff,
                                    "train_Pitch_loss":Pitch_diff, 
                                    "train_Roll_loss":Roll_diff,
                                    }, step=i + epoch * total_data_loaded)

                save_pcd('./result/predicted_pcd.pcd', genData['pcd']['pred'][0].detach().cpu().numpy())
                save_pcd('./result/expected_pcd.pcd' , genData['pcd']['exp'][0].detach().cpu().numpy())

                print(f'[Epoch: {epoch + 1}, Batch: {i + 1} / {total_data_loaded}], Total loss {running_loss}')
                running_loss = 0.0

        scheduler.step()
        if epoch % args.save_ckp_freq == 0: accelerator.save_state() 

    if args.logging_type is not None :
        accelerator.end_training()

if __name__ == "__main__":
    args = config.get_parser(config_file)

    if args.multi_gpu_tr == True:
        transformer_auto_wrapper_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                EncoderBlock,
            },
        )
        fsdp_plugin = FullyShardedDataParallelPlugin(
            # state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            # optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            state_dict_config = ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True),
            auto_wrap_policy = transformer_auto_wrapper_policy,
            sharding_strategy = ShardingStrategy.FULL_SHARD,
            mixed_precision_policy = MixedPrecision(reduce_dtype =torch.float16),
        )

        # Initialize accelerator
        project_config = ProjectConfiguration(project_dir=args.prj_dir, automatic_checkpoint_naming=True)
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin, project_config=project_config, log_with=args.logging_type)
        print('Check plugin: ', accelerator.state.fsdp_plugin)
        
        model = TransformerCalib(device=accelerator.device, args=args)
        device = accelerator.device
        # model = accelerator.prepare_model(model)

    else:
        num_gpus = torch.cuda.device_count()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = TransformerCalib(device=device, args=args)
        model.to(device=device)

        if args.data_parallel == True:
            model = torch.nn.parallel.DataParallel(
                model, device_ids=list(range(num_gpus)), dim=0)

    dataSet = PreKittiData(root_dir=args.data_root, args=args)
    valid_loader = DataLoader(dataSet.getData(valid=False), batch_size=args.batch_size, drop_last=True, num_workers=os.cpu_count()//2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.sche_step_size, gamma=args.sche_gamma)
    writer = SummaryWriter(args.save_writter_path)

    if args.multi_gpu_tr == True:
        model, optimizer, valid_loader, scheduler = accelerator.prepare(model, optimizer, valid_loader, scheduler)
        config.result_dir_preparation(args, accelerator.process_index)
        accelerator.wait_for_everyone()
        print('Node', accelerator.process_index , 'Start trainning...')
    else:
        print('Start trainning...')

    train(model=model, train_loader=valid_loader, device=device, 
        optimizer=optimizer, writer=writer, scheduler=scheduler, args=args)

    if args.multi_gpu_tr == True: print('Node', accelerator.process_index , 'Success')
    else: print('Success')

    writer.close()