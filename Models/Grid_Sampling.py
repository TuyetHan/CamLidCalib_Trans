import torch
from torch_geometric.nn import voxel_grid

def grid_sample(pos, batch_ind, size, start, max_pc_voxel):
    """
    Args:
      pos of shape (n_point*batch_size, 3): Point cloud of all batch (float)
      batch_ind (n_point *batch_size): Tensor indicate which batch that point belong to.
      size: Size of a voxel
      start: Start coordinates of the grid
      max_pc_voxel (int): Maximum number of point cloud within one voxel

    Returns:
      z is a list of pad tensor of shape (batch_size, max_pc_voxel, 3): Encoded input sequences.
    """

    # Initilization
    batch_size = batch_ind.max().item() + 1
    all_clusters_points = [[] for _ in range(batch_size)]
    list_batch_cluster = []

    # Clustering: Cluster is idx of each point in voxel
    cluster = voxel_grid(pos = pos, batch = batch_ind, size = size, start=start) #[N, ]
    unique_voxels, counts = cluster.unique(return_counts=True)

    # Create a list of tensor cluster of each batch
    voxel_batches = torch.zeros_like(unique_voxels)
    for i, voxel in enumerate(unique_voxels):
        voxel_mask = (cluster == voxel)
        batches = batch_ind[voxel_mask]
        voxel_batches[i] = batches[0]
        assert batches.unique().numel() == 1, "A voxel contains points from different batches"
        points_in_cluster = pos[voxel_mask.view(-1)]

        # If the cluster has fewer points than `max_points`, pad it
        if points_in_cluster.shape[0] < max_pc_voxel:
            padded_points = torch.cat([points_in_cluster, torch.zeros(max_pc_voxel - points_in_cluster.shape[0], 3).to(points_in_cluster.device)], dim=0)
        # If the cluster has more points than `max_pc_voxel`, trim it
        elif points_in_cluster.shape[0] > max_pc_voxel:
            padded_points = points_in_cluster[:max_pc_voxel]
        # If the cluster has exactly `max_points`, use it as is
        else:
            padded_points = points_in_cluster
        all_clusters_points[batches[0]].append(padded_points)

    max_n_clusters = torch.max(torch.bincount(voxel_batches))
    for i in range(max_n_clusters):
      # batch_cluster = [(sublist[i]) for sublist in all_clusters_points]
      batch_cluster = [(sublist[i] if i < len(sublist) else torch.zeros((max_pc_voxel, 3)).to(points_in_cluster.device)) for sublist in all_clusters_points]
      batch_cluster = torch.stack(batch_cluster)
      list_batch_cluster.append(batch_cluster)

    # return voxel_batches, unique_voxels, counts, all_clusters_points, list_batch_cluster
    return list_batch_cluster