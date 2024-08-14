
import time
import numpy as np
import open3d as o3d


import numpy as np
import open3d as o3d
import time

def pointcloud_to_voxelgrid_optimized(voxel_size, center, height, width, depth):
    """
    Optimized version: Creates a voxel grid from a point cloud where each point represents a voxel.
    The point's color encodes its voxel index.

    Args:
        voxel_size: The size of each voxel.
        center: The center of the voxel grid.
        height, width, depth: The dimensions of the voxel grid in meters.

    Returns:
        An Open3D voxel grid.
    """

    # Round dimensions to the next multiple of voxel_size
    height = np.ceil(height / voxel_size) * voxel_size
    width = np.ceil(width / voxel_size) * voxel_size
    depth = np.ceil(depth / voxel_size) * voxel_size

    # Calculate number of voxels along each dimension
    num_voxels_height = int(height / voxel_size)
    num_voxels_width = int(width / voxel_size)
    num_voxels_depth = int(depth / voxel_size)

    # Calculate voxel grid bounds
    half_height = height / 2
    half_width = width / 2
    half_depth = depth / 2
    min_bound = center - np.array([half_width, half_height, half_depth])
    print(min_bound)

    # Create grid of indices using NumPy's meshgrid
    i, j, k = np.mgrid[:num_voxels_height, :num_voxels_width, :num_voxels_depth]

    # Calculate point positions using vectorized operations
    points = min_bound + np.stack([j, i, k], axis=-1) * voxel_size
    points = points.reshape(-1, 3)  # Flatten into a list of points

    # Calculate colors using vectorized operations and normalize
    colors = np.stack([i, j, k], axis=-1) / np.array([num_voxels_height, num_voxels_width, num_voxels_depth])
    colors = colors.reshape(-1, 3)  # Flatten into a list of colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)  


    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)  


    return voxel_grid

# Example usage (adjusting height, width, depth to meters)
voxel_size = 0.01
center = np.array([0.5, 0.5, 0.5])
height, width, depth = 1.0, 1.0, 1.0  # In meters
print("instanciate grid")
start_time = time.time()
voxel_grid = pointcloud_to_voxelgrid_optimized(voxel_size, center, height, width, depth)
print(f"Execution time: {time.time() - start_time:.2f} seconds")
# Visualize the voxel grid (optional)

voxel_grid_should_be_same = o3d.geometry.VoxelGrid.create_dense(
    origin=center,
    voxel_size=voxel_size,
    width=width,
    height=height,
    depth=depth,
    color=[0, 0, 0]
)

o3d.visualization.draw_geometries([voxel_grid, voxel_grid_should_be_same])