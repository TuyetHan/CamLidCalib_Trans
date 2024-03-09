import torch
from torch_geometric.nn import voxel_grid

def grid_sample(pos, feature, batch_ind, size, start):
    """
    Args:
      pos of shape (n_point*batch_size, 3): Point cloud of all batch (float)
      feature of shape (n_point*batch_size, M): feature of pc(Ref, RGB or from KPConv)
      batch_ind (n_point *batch_size, 1): Tensor indicate which batch that point belong to.
      size: Size of a voxel
      start: Start coordinates of the grid

    Returns:
      all_clusters_points is a 2D list [B, C] of cluster tensor of shape (P, M+3)
        B: Batch size
        C: Number of clusters in each batch (vary depend on voxel size and point cloud density)
        P: Number of points in each cluster (vary depend on voxel size and point cloud density)
        M + 3:  Number of features + 3D position
      max_n_clusters is  maximum of C
    """
    # Initilization
    batch_size = batch_ind.max().item() + 1
    all_clusters_points = [[] for _ in range(batch_size)]
    # list_batch_cluster = []

    # Clustering: Cluster is idx of each point in voxel
    cluster = voxel_grid(pos = pos, batch = batch_ind, size = size, start=start) #[N, ]
    unique_voxels, counts = cluster.unique(return_counts=True, sorted = False)

    # Stack Position and Feature together
    point_wfeature = torch.hstack((pos, feature))

    # Create a list of tensor cluster of each batch
    voxel_batches = torch.zeros_like(unique_voxels)
    max_n_clusters = torch.max(torch.bincount(voxel_batches))
    for i, voxel in enumerate(unique_voxels):
        voxel_mask = (cluster == voxel)
        batches = batch_ind[voxel_mask]
        voxel_batches[i] = batches[0]
        assert batches.unique().numel() == 1, "A voxel contains points from different batches"
        points_in_cluster = point_wfeature[voxel_mask.view(-1)]
        all_clusters_points[batches[0]].append(points_in_cluster)

    # return voxel_batches, unique_voxels, counts, all_clusters_points
    return all_clusters_points, max_n_clusters