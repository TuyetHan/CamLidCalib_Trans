import os
import argparse
import functools
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
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch, MixedPrecision, ShardingStrategy, StateDictType

def get_parser():
    parser = argparse.ArgumentParser(description='Camera Lidar Calibration using Transformer Network')
    parser.add_argument('--config', type=str, default='CamLidCalib_Trans/config/Trans_Calib_1st.yaml', help='config file')
    parser.add_argument('opts', help='see config/Trans_Calib_1st.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def train(model=None, train_loader:DataLoader=None, device:torch.device='cuda', 
          save:bool=True, writer:SummaryWriter=None, epochs:int=100, optimizer:torch.optim.Adam=None, 
          scheduler:torch.optim.lr_scheduler.StepLR=None, minLoss=torch.inf, args:argparse.Namespace=None):
    
    total_data_loaded = len(train_loader)
    print_freq = args.print_log_freq
    All_batch_size = args.batch_size

    correspondance_point_loss = nn.MSELoss(reduction='none')
    num_recursive_iter = args.num_recursive_iter
    epochs = args.epochs

    model.train()

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
            if i % print_freq == 0:    # print every 10 sample
                pred_tran = resT[:, :3, 3].detach().cpu()
                exp_trans = -transform[:, :3, 3].detach().cpu()
                pred_rot = resT[:, :3, :3].detach().cpu()
                exp_rot = transform[:, :3, :3].permute(0, 2, 1).detach().cpu()

                running_loss /= print_freq
                geodesic_dist = rotationLoss(pred_rot, exp_rot)
                X_diff, Y_diff, Z_diff = translaitionLoss(pred_tran, exp_trans)
                Yaw_diff, Pitch_diff, Roll_diff = anglesLoss(pred_rot, exp_rot)

                if writer:
                    writer.add_scalar('Loss/Train', running_loss, i + epoch * total_data_loaded)
                    writer.add_scalar('Loss_Geodesic/Train', geodesic_dist, i + epoch * total_data_loaded)
                    writer.add_scalar('Loss_X/Train', X_diff, i + epoch * total_data_loaded)
                    writer.add_scalar('Loss_Y/Train', Y_diff, i + epoch * total_data_loaded)
                    writer.add_scalar('Loss_Z/Train', Z_diff, i + epoch * total_data_loaded)
                    writer.add_scalar('Loss_Yaw/Train', Yaw_diff, i + epoch * total_data_loaded)
                    writer.add_scalar('Loss_Pitch/Train', Pitch_diff, i + epoch * total_data_loaded)
                    writer.add_scalar('Loss_Roll/Train', Roll_diff, i + epoch * total_data_loaded)

                save_pcd('./result/predicted_pcd.pcd', genData['pcd']['pred'][0].detach().cpu().numpy())
                save_pcd('./result/expected_pcd.pcd', genData['pcd']['exp'][0].detach().cpu().numpy())

                print(f'[Epoch: {epoch + 1}, Batch: {i + 1} / {total_data_loaded}], Total loss {running_loss}')
                running_loss = 0.0

        scheduler.step()

        epochLosses /= (i+1)
        if save and minLoss > epochLosses:
            minLoss = epochLosses
            print(f"saving model with mean Loss {minLoss}")
            torch.save({
                'epoch' : epoch,
                'minLoss': minLoss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join('./result/trained_model', "save_"+str(epoch)+".pth"))
        

if __name__ == "__main__":
    args = get_parser()

    if args.multi_gpu_tr == True:
        transformer_auto_wrapper_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                EncoderBlock,
            },
        )
        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy = transformer_auto_wrapper_policy,
            sharding_strategy = ShardingStrategy.FULL_SHARD,
            mixed_precision_policy = MixedPrecision(reduce_dtype =torch.float16),
        )

        # Initialize accelerator
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
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
        print('Node', accelerator.process_index , 'Start trainning...')
    else:
        print('Start trainning...')

    train(model=model, train_loader=valid_loader, device=device, 
        optimizer=optimizer, writer=writer, scheduler=scheduler, args=args)

    if args.multi_gpu_tr == True: print('Node', accelerator.process_index , 'Success')
    else: print('Success')

    writer.close()