
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
    
    start_time = time.time()

    # Calculate number of voxels along each dimension
    num_voxels_height = int(np.ceil(height / voxel_size))
    num_voxels_width = int(np.ceil(width / voxel_size))
    num_voxels_depth = int(np.ceil(depth / voxel_size))
    print(f"Step 1 (Calculate number of voxels): {time.time() - start_time:.4f} seconds")

    # Calculate voxel grid bounds (adjusting for potential rounding in num_voxels)
    step_time = time.time()
    half_height = num_voxels_height * voxel_size / 2
    half_width = num_voxels_width * voxel_size / 2
    half_depth = num_voxels_depth * voxel_size / 2
    min_bound = center - np.array([half_width, half_height, half_depth])
    print(f"Step 2 (Calculate voxel grid bounds): {time.time() - step_time:.4f} seconds")

    # Create grid of indices using NumPy's meshgrid
    step_time = time.time()
    i, j, k = np.mgrid[:num_voxels_height, :num_voxels_width, :num_voxels_depth]
    print(f"Step 3 (Create grid of indices): {time.time() - step_time:.4f} seconds")

    # Calculate point positions using vectorized operations
    step_time = time.time()
    points = min_bound + np.stack([j, i, k], axis=-1) * voxel_size
    points = points.reshape(-1, 3)  # Flatten into a list of points
    print(f"Step 4 (Calculate point positions): {time.time() - step_time:.4f} seconds")

    # Calculate colors using vectorized operations and normalize
    step_time = time.time()
    colors = np.stack([i, j, k], axis=-1) / np.array([num_voxels_height, num_voxels_width, num_voxels_depth])
    colors = colors.reshape(-1, 3)  # Flatten into a list of colors
    print(f"Step 5 (Calculate colors): {time.time() - step_time:.4f} seconds")

    # Create point cloud
    step_time = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Step 6 (Create point cloud): {time.time() - step_time:.4f} seconds")

    # Create voxel grid from point cloud
    step_time = time.time()
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    print(f"Step 7 (Create voxel grid): {time.time() - step_time:.4f} seconds")

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.4f} seconds")

    return voxel_grid

# Example usage (adjusting height, width, depth to meters)
voxel_size = 0.0025
center = np.array([0.5, 0.5, 0.5])
height, width, depth = 1.0, 1.0, 1.0  # In meters
print("instanciate grid")
start_time = time.time()
voxel_grid = pointcloud_to_voxelgrid_optimized(voxel_size, center, height, width, depth)
print(f"Execution time: {time.time() - start_time:.2f} seconds")
# Visualize the voxel grid (optional)
o3d.visualization.draw_geometries([voxel_grid])