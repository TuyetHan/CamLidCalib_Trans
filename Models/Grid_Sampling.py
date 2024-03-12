import torch
from torch_geometric.nn import voxel_grid

def grid_sample(pos, batch_ind, size, start):
    """
    Args:
      pos of shape (n_point*batch_size, 3): Point cloud of all batch (float)
      batch_ind (n_point *batch_size): Tensor indicate which batch that point belong to.
      size: Size of a voxel
      start: Start coordinates of the grid

    Returns:
      cluster mask of shape (batch_size, n_point, n_point): 
        Attention Cluster Mask, only caculate attention bw 2 points in the same cluster.
    """
    # Initilization
    batch_size = batch_ind.max().item() + 1
    n_point = pos.shape[0]//batch_size
    all_cluster_mask = [torch.zeros(n_point, n_point).to(pos.device) for _ in range(batch_size)]

    # Clustering: Cluster is idx of each point in voxel
    cluster = voxel_grid(pos = pos, batch = batch_ind, size = size, start=start)
    unique_voxels, counts = cluster.unique(return_counts=True, sorted = False)

    # Create a list of tensor cluster of each batch
    for i, voxel in enumerate(unique_voxels):
        # Find batch id of current voxel
        voxel_mask = (cluster == voxel)
        batch_id = batch_ind[voxel_mask][0]

        # Only take points in the same batch
        batch_mask = (batch_ind == batch_id)
        batch_voxel_mask = voxel_mask.unsqueeze(1)[batch_mask]
        if len(batch_voxel_mask) != n_point:
            raise ImportError('Not yet support different size pointcloud. All PC should have same size!!')

        # Create 2D mask and merge with all cluster mask
        mask_2d = torch.outer(batch_voxel_mask, batch_voxel_mask)
        all_cluster_mask[batch_id] = torch.logical_or(all_cluster_mask[batch_id], mask_2d)

    all_cluster_mask = torch.stack(all_cluster_mask)
    return all_cluster_mask