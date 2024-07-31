import numpy as np
import open3d as o3d
from copy import deepcopy
from pathlib import Path
import cv2
from utils.annotationimage import AnnotationImage
from utils.voxelgrid import VoxelGrid
import utils.vis_masks as vis_masks

class AnnotationScene:
    def __init__(self, scene_id, scene_reader, camera_intrinsics, img_width, img_height):
        self.scene_id = scene_id
        self.scene_reader = scene_reader
        self.annotation_images = {}
        self.active_image = None
        self.voxel_grid = None
        self.camera_intrinsics = camera_intrinsics
        self.img_width = img_width
        self.img_height = img_height
        self.poi = self.get_cameras_point_of_interest()
        self.dataset_object_ids, self.names, self.scene_object_ids = self.get_object_metadata()
        # -> list object names in the beginning. let annotater add to the list object name and count 

    def get_object_metadata(self):
        objects_data = self.scene_reader.get_object_poses(self.scene_id)
        object_list = [inner_list[0] for inner_list in objects_data]
        dataset_object_ids = []
        names = []
        scene_object_ids = []
        for i in range(len(object_list)):
            dataset_object_ids.append(object_list[i].id)
            names.append(object_list[i].name)
            scene_object_ids.append(i+1)
        return dataset_object_ids, names, scene_object_ids

    def greedy_farthest_point_reorder(points, start_index=0):

        num_points = points.shape[0]
        remaining_indices = list(range(num_points))  # Indices of points not yet chosen
        reordered_indices = [start_index]  # Start with the specified index

        # Remove the starting point from remaining indices
        remaining_indices.remove(start_index)
        current_point = points[start_index]

        # Greedily choose farthest points
        while remaining_indices:
            distances = np.linalg.norm(points[remaining_indices] - current_point, axis=1)  # Calculate distances
            farthest_index = remaining_indices[np.argmax(distances)]  # Index of farthest point

            reordered_indices.append(farthest_index)
            remaining_indices.remove(farthest_index)
            current_point = points[farthest_index]

        return reordered_indices

    def load_images(self):
        image_paths = self.scene_reader.get_images_rgb_path(self.scene_id)
        camera_poses = self.scene_reader.get_camera_poses(self.scene_id)
        rigit_segmaps = self.get_rigit_segmaps()

        #perform smart camera pose ordering
        reordering = self.max_distance_camera_reorder(camera_poses, k=int(len(camera_poses)/2))
        # camera_poses = [camera_poses[i] for i in reordering]
        # image_paths = [image_paths[i] for i in reordering]
        # rigit_silhouette = rigit_silhouette[reordering]
        # for i, image_path in enumerate(image_paths):
        #     image_path = Path(image_path)
        #     self.annotation_images[image_path.name] = AnnotationImage(image_path, camera_poses[i], rigit_silhouette[i])
        for i in reordering:
            image_path = Path(image_paths[i])
            self.annotation_images[image_path.name] = AnnotationImage(image_path, camera_poses[i], rigit_segmaps[i])
    
    def get_rigit_segmaps(self):
        # read in all scene masks

        # mask count
        mask_path = Path(self.scene_reader.get_images_mask_path(self.scene_id))

        pose_count = len(self.scene_reader.get_camera_poses(self.scene_id))
        object_count = len(self.scene_reader.get_object_poses(self.scene_id))
        width = self.scene_reader.get_camera_info_scene(self.scene_id).width
        height = self.scene_reader.get_camera_info_scene(self.scene_id).height
        mask_count = pose_count * object_count
        masks = np.zeros((mask_count, height, width), dtype=np.uint8)
        for scene_object_id, object_name in zip(self.scene_object_ids, self.names):
            for j in range(pose_count):
                mask_file = mask_path / f"{object_name}_{scene_object_id-1:03d}_{j+1:06d}.png"
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                masks[(scene_object_id-1) * pose_count + j][mask > 0] = scene_object_id

        masks_reshaped = masks.reshape(object_count, pose_count, height, width)
        segmaps = np.sum(masks_reshaped, axis=0).astype(np.uint8)

        return segmaps   

    def generate_masks(self):
        vis_masks.create_masks(self.scene_reader, self.scene_id)

    def get_images(self):
        return self.annotationImages.keys()
    
    def get_image_paths(self):
        return [Path(image_path) for image_path in self.scene_reader.get_images_rgb_path(self.scene_id)]
    
    def has_correct_number_of_masks(self):
        masks_dir = Path(self.scene_reader.get_images_mask_path(self.scene_id))
        if not masks_dir.exists(): 
            return False
        else:
            expected_mask_count = len(self.scene_reader.get_object_poses(self.scene_id)) * len(self.scene_reader.get_camera_poses(self.scene_id))
            if expected_mask_count != len(list(masks_dir.iterdir())):
                return False
            return True
        
    def max_distance_camera_reorder(self, poses, k=7, start_index=0):
        points = np.array([pose.translation() for pose in poses])

        # Greedy reordering for the first k positions, considering all points and maximizing distance from selected points
        farthest_indices = []
        remaining_indices = list(range(len(points)))
        current_index = start_index
        for _ in range(k):
            remaining_indices.remove(current_index)
            farthest_indices.append(current_index)
            # Calculate distances from all remaining points to ALL previously selected points
            distances_to_selected = np.linalg.norm(
                points[remaining_indices][:, np.newaxis] - points[farthest_indices], axis=2
            )
            # Find the minimum distance to any selected point for each remaining point
            min_distances_to_selected = np.min(distances_to_selected, axis=1)
            # Select the point with the maximum minimum distance
            current_index = remaining_indices[np.argmax(min_distances_to_selected)]

        # Remaining indices in original order
        remaining_indices = [i for i in range(len(poses)) if i not in farthest_indices]

        # Combine the indices
        reordered_indices = farthest_indices + remaining_indices

        return reordered_indices


        # Combine and Return Indices
        remaining_indices = [i for i in range(len(poses)) if i not in downsampled_indices]
        reordered_indices = downsampled_indices + remaining_indices 
        print(reordered_indices)
        return reordered_indices
    
    def instanciate_voxel_grid_at_poi(self, voxel_size=0.005):

        #TODO maybe add warning if obejcts are further away from the poi than 0.5m

        camera_poses = [image.camera_pose for image in self.annotation_images.values()]

        # get largest distance between cameras and ray intersection
        distances = []
        for camera_pose in camera_poses:
            distances.append(np.linalg.norm(self.poi - camera_pose.tf[:3, 3]))
        max_distance = np.max(distances)

        width = max_distance * 1.5
        height = max_distance * 1.5
        depth = max_distance * 1.5
        print(width, height, depth)

        self.voxel_grid = VoxelGrid(
            width=width,
            height=height,
            depth=depth,
            voxel_size=voxel_size,
            origin=np.array([self.poi[0] - width/2, self.poi[1] - height/2, self.poi[2] - depth/2]).astype(np.float64),
            color=np.array([0.2, 0.2, 0.2]).astype(np.float64)
        )

        #visualize voxel grid
        # o3d.visualization.draw_geometries([self.voxel_grid])

        print("prefiltering")

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.img_width,
            self.img_height,
            self.camera_intrinsics[0, 0], 
            self.camera_intrinsics[1, 1], 
            self.camera_intrinsics[0, 2], 
            self.camera_intrinsics[1, 2])

        #get all camera poses of images that have accepted annotations
        images = [image for image in self.annotation_images.values() if image.annotation_accepted]
        camera_poses = [image.camera_pose for image in images]

        relevant_points = []
        for pose in camera_poses:
            pose = pose.tf
            mask_grid = deepcopy(self.voxel_grid.o3d_grid)
            mask = np.ones((self.img_height, self.img_width))
            silhouette = o3d.geometry.Image(mask.astype(np.float32))
            extrinsic = np.linalg.inv(pose)
            intrinsic = camera_intrinsics
            
            cam = o3d.camera.PinholeCameraParameters()
            cam.intrinsic = intrinsic
            cam.extrinsic = extrinsic
            mask_grid.carve_silhouette(silhouette, cam, keep_voxels_outside_image=False)
            relevant_points += [voxel.grid_index for voxel in mask_grid.get_voxels()]
            print(mask_grid)

        for voxel in self.voxel_grid.o3d_grid.get_voxels(): #TODO why not use clear()?
            self.voxel_grid.o3d_grid.remove_voxel(voxel.grid_index)
        for pos in relevant_points:
            new_voxel = o3d.geometry.Voxel(pos, [pos[0]/(width/voxel_size), pos[1]/(height/voxel_size), pos[2]/(depth/voxel_size)])
            self.voxel_grid.o3d_grid.add_voxel(new_voxel)

        for pose in camera_poses:
            pose = pose.tf
            mask_grid = deepcopy(self.voxel_grid.o3d_grid)
            mask = np.zeros((self.img_height, self.img_width))
            silhouette = o3d.geometry.Image(mask.astype(np.float32))
            extrinsic = np.linalg.inv(pose)
            intrinsic = camera_intrinsics
            
            cam = o3d.camera.PinholeCameraParameters()
            cam.intrinsic = intrinsic
            cam.extrinsic = extrinsic
            mask_grid.carve_silhouette(silhouette, cam, keep_voxels_outside_image=True)

        for voxel in mask_grid.get_voxels():
            self.voxel_grid.o3d_grid.remove_voxel(voxel.grid_index)

        for image in images:
            self.carve_silhouette(image, keep_voxels_outside_image=True)

    def get_cameras_point_of_interest(self, debug_vizualization=False):
        '''adepted from https://math.stackexchange.com/questions/4865611/intersection-closest-point-of-multiple-rays-in-3d-space'''

        camera_poses = self.scene_reader.get_camera_poses(self.scene_id)

        vis = [pose.tf[:3, 2] for pose in camera_poses]
        ois = [pose.tf[:3, 3] for pose in camera_poses]

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
            self.visualize_rays_and_intersection(ois, vis, x1)
        return x1
    
    def carve_silhouette(self, image, keep_voxels_outside_image):
        mask = image.segmap.astype(np.uint8)
        for obj in image.annotation_objects.values():
            if obj.mask is None:
                continue
            mask += obj.mask.astype(np.uint8)
        mask[mask > 0] = 1
        print("debug1")

        extrinsic = np.linalg.inv(image.camera_pose.tf)
        silhouette = o3d.geometry.Image(mask.astype(np.float32))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.img_width, self.img_height, self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1], self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2])

        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = intrinsic
        cam.extrinsic = extrinsic

        self.voxel_grid.o3d_grid.carve_silhouette(silhouette, cam, keep_voxels_outside_image=keep_voxels_outside_image)
        
    def visualize_rays_and_intersection(self, ois, vis, intersection_point):
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

    def get_random_color(self):
        return list(np.random.choice(range(256), size=3) / 255.0)
