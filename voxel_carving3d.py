import open3d as o3d
import numpy as np
from utils.vis_masks import SceneFileReader
from pathlib import Path
import cv2
import time
import random

def get_rigit_silhouette(scene):
    # read in all scene masks
    mask_path = Path(scene.root_dir) / scene.scenes_dir / scene_id / scene.mask_dir
    # mask count
    pose_count = len(scene.get_camera_poses(scene_id))
    object_count = len(scene.get_object_poses(scene_id))
    width = scene.get_camera_info_scene(scene_id).width
    height = scene.get_camera_info_scene(scene_id).height
    mask_count = pose_count * object_count
    masks = np.zeros((mask_count, height, width), dtype=np.uint8)
    i = 0
    for mask_file in sorted(mask_path.iterdir()):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        masks[i] = mask
        i += 1

    masks_reshaped = masks.reshape(object_count, pose_count, height, width)
    rigid_silhouettes = np.sum(masks_reshaped, axis=0).astype(np.uint8)
    rigid_silhouettes[rigid_silhouettes > 0] = 255
    return rigid_silhouettes   

def visualize_rays_and_intersection(ois, vis, intersection_point):
    geometries = []

    # Visualize rays
    for oi, vi in zip(ois, vis):
        line_points = [oi, oi + vi * 10]  # Extend the direction vector for visualization
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color
        geometries.append(line)
    
    # Visualize intersection point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.paint_uniform_color([0, 1, 0])  # Green color
    sphere.translate(intersection_point)
    geometries.append(sphere)
    
    o3d.visualization.draw_geometries(geometries)

def determine_intersection_of_rays(camera_extrinsics, debug_vizualization=False):
    '''adepted from https://math.stackexchange.com/questions/4865611/intersection-closest-point-of-multiple-rays-in-3d-space'''
    
    vis = [pose[:3, 2] for pose in camera_extrinsics]
    ois = [pose[:3, 3] for pose in camera_extrinsics]

    q = np.zeros((3, 3))
    b = np.zeros(3)
    c = 0
    for oi, vi in zip(ois, vis):
        p0 = np.eye(3) - np.outer(vi, vi)
        q += p0
        poi = np.dot(p0, oi) * -2
        b += poi
        c += np.dot(oi, oi)

    try:
        qinv = np.linalg.inv(q)
    except np.linalg.LinAlgError:
        print("Matrix not invertible")
        return None

    x1 = np.dot(qinv, b) * -0.5
    if debug_vizualization:
        visualize_rays_and_intersection(ois, vis, x1)
    return x1

if __name__ == "__main__":

    DATASET_PATH = Path('../depth-estimation-of-transparent-objects')
    scene_id = 'j_005'
    scene = SceneFileReader.create(DATASET_PATH / 'config.cfg')
    silhouettes = get_rigit_silhouette(scene)

    # calc obejct poses center
    object_poses = scene.get_object_poses(scene_id)
    translations = []
    for object_pose in object_poses:
        translations.append(object_pose[1][3::4][:3])
    obj_centers = np.mean(translations, axis=0)

    # calc camera poses center
    camera_poses = scene.get_camera_poses(scene_id)
    camera_extrinsics = [camera_pose.tf for camera_pose in camera_poses]
    camera_poi = determine_intersection_of_rays(camera_extrinsics)

    #calc distance between object and camera center
    distance = np.linalg.norm(obj_centers - camera_poi)
    if distance >= 0.5:
        print(f"WARNING: Distance between object and camera center is {distance}m.")
        print(f"Objects should be closer to the camera center.")

    #calc largest distance between cameras and ray intersection
    distances = []
    for camera_pose in camera_extrinsics:
        distances.append(np.linalg.norm(camera_poi - camera_pose[:3, 3]))
    max_distance = np.max(distances)

    width = max_distance * 1.5
    height = max_distance * 1.5
    depth = max_distance * 1.5
    voxel_size = 0.005

    voxel_carving_grid = o3d.geometry.VoxelGrid.create_dense(
        width=width,
        height=height,
        depth=depth,
        voxel_size=voxel_size,
        origin=[camera_poi[0] - width/2, camera_poi[1] - height/2, camera_poi[2] - depth/2],
        color=[0.2, 0.2, 0.2]
    )

    for mask, pose in zip(silhouettes, camera_poses):
        # show mask
        silhouette = o3d.geometry.Image(mask.astype(np.float32))
        extrinsic = np.linalg.inv(pose.tf)
        intrinsic = scene.get_camera_info_scene(scene_id).as_o3d()

        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = intrinsic
        cam.extrinsic = extrinsic

        voxel_carving_grid.carve_silhouette(silhouette, cam, keep_voxels_outside_image=True)
        print(voxel_carving_grid)
        
    o3d.visualization.draw_geometries([voxel_carving_grid])

