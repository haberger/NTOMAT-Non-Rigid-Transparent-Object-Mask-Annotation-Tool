import numpy as np
import open3d as o3d
from copy import deepcopy
from pathlib import Path
import cv2
from utils.annotationimage import AnnotationImage, AnnotationObject
from utils.voxelgrid import VoxelGrid
import utils.vis_masks as vis_masks
import pickle
import time
# from utils.write_3ddat_to_bop import write_scene_to_bop
import json
from tqdm import tqdm 
from utils.scenerenderer import SceneRenderer, get_bbox_from_mask
import trimesh
import os

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
        self.upside_down = False

        self.poi = None
        self.dataset_object_ids = None
        self.names = None
        self.scene_object_ids = None

        self.manual_annotation_done = False
        self.visualization_opacity = 0.5

    def load_scene_data(self):
        self.poi = self.get_cameras_point_of_interest()
        self.dataset_object_ids, self.names, self.scene_object_ids = self.get_object_metadata()

    def scene_to_pickle(self, folder="../debug_data_promptgeneration"):
        path = Path(folder)
        with open(path/"scene_id.pkl", "wb") as f:
            pickle.dump(self.scene_id, f)
        with open(path/"scene_reader.pkl", "wb") as f:
            pickle.dump(self.scene_reader, f)
        with open(path/"camera_intrinsics.pkl", "wb") as f:
            pickle.dump(self.camera_intrinsics, f)
        with open(path/"img_width.pkl", "wb") as f:
            pickle.dump(self.img_width, f)
        with open(path/"img_height.pkl", "wb") as f:
            pickle.dump(self.img_height, f)
        with open(path/"poi.pkl", "wb") as f:
            pickle.dump(self.poi, f)
        with open(path/"dataset_object_ids.pkl", "wb") as f:
            pickle.dump(self.dataset_object_ids, f)
        with open(path/"names.pkl", "wb") as f:
            pickle.dump(self.names, f)
        with open(path/"scene_object_ids.pkl", "wb") as f:
            pickle.dump(self.scene_object_ids, f)
        with open(path/"annotation_images.pkl", "wb") as f:
            pickle.dump(self.annotation_images, f)
        with open(path/"active_image.pkl", "wb") as f:
            pickle.dump(self.active_image, f)
        
        self.voxel_grid.save_voxelgrid(path)

    def scene_from_pickle(self, folder="../debug_data_promptgeneration"):
        path = Path(folder)
        with open(path/"scene_id.pkl", "rb") as f:
            self.scene_id = pickle.load(f)
        with open(path/"scene_reader.pkl", "rb") as f:
            self.scene_reader = pickle.load(f)
        with open(path/"camera_intrinsics.pkl", "rb") as f:
            self.camera_intrinsics = pickle.load(f)
        with open(path/"img_width.pkl", "rb") as f:
            self.img_width = pickle.load(f)
        with open(path/"img_height.pkl", "rb") as f:
            self.img_height = pickle.load(f)
        with open(path/"poi.pkl", "rb") as f:
            self.poi = pickle.load(f)
        with open(path/"dataset_object_ids.pkl", "rb") as f:
            self.dataset_object_ids = pickle.load(f)
        with open(path/"names.pkl", "rb") as f:
            self.names = pickle.load(f)
        with open(path/"scene_object_ids.pkl", "rb") as f:
            self.scene_object_ids = pickle.load(f)
        with open(path/"annotation_images.pkl", "rb") as f:
            self.annotation_images = pickle.load(f)
        with open(path/"active_image.pkl", "rb") as f:
            self.active_image = pickle.load(f)
        
        self.voxel_grid = VoxelGrid()
        self.voxel_grid.load_voxelgrid(path)

    def add_object(self, object_name, object_dataset_id):
        radio_options = [obj.label for obj in self.active_image.annotation_objects.values()]

        i=0
        object_name = f"{object_name}_{i}"
        while object_name in radio_options:
            i+=1
            object_name = f"{object_name.rpartition('_')[0]}_{i}"

        scene_object_id = max(self.scene_object_ids)+1
        # self.scene_object_ids.append(scene_object_id)

        for anno_image in self.annotation_images.values():

            annotation_object = AnnotationObject([], [], None, None, object_name, object_dataset_id, scene_object_id)
            anno_image.active_object = annotation_object #TODO check if this handles corrrectly if i change image in the middle of the annotation process
            anno_image.annotation_objects[annotation_object.label] = annotation_object
        
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
    
    def get_fully_visible_objects_from_segmap(self, segmap, scene_object_ids):
        unique_ids = np.unique(segmap)
        unique_ids = unique_ids[unique_ids != 0]

        border_ids = np.unique(
            np.concatenate((
                np.unique(segmap[0, :]),
                np.unique(segmap[-1, :]),
                np.unique(segmap[:, 0]),
                np.unique(segmap[:, -1]))
            )
        )
        fully_visible_ids = np.zeros_like(scene_object_ids).astype(bool)
        for i, id in enumerate(scene_object_ids):
            if id in unique_ids and id not in border_ids:
                fully_visible_ids[i] = True
                
        return fully_visible_ids

    def find_best_segmap(self, segmaps):
        best_segmap_index = None
        max_objects = 0

        for index, segmap in enumerate(segmaps):  # Enumerate to get indices
            fully_visible_ids = self.get_fully_visible_objects_from_segmap(segmap, self.scene_object_ids)

            if np.all(fully_visible_ids):
                return index

            if sum(fully_visible_ids) > max_objects:
                best_segmap_index = index
                max_objects = sum(fully_visible_ids)
                
        return best_segmap_index
    
    def load_images(self):
        image_paths = self.scene_reader.get_images_rgb_path(self.scene_id)
        camera_poses = self.scene_reader.get_camera_poses(self.scene_id)
        rigit_segmaps = self.get_rigit_segmaps()

        best_segmap_index = self.find_best_segmap(rigit_segmaps)

        #perform smart camera pose ordering
        reordering = self.max_distance_camera_reorder(camera_poses, k=int(len(camera_poses)/2), start_index=best_segmap_index)

        for i in reordering:
            image_path = Path(image_paths[i])
            self.annotation_images[image_path.name] = AnnotationImage(image_path, camera_poses[i], self, rigit_segmaps[i])

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

    
    def instanciate_voxel_grid_at_poi_fast(self, trigger_image, voxel_size=0.005,):

        camera_poses = [image.camera_pose for image in self.annotation_images.values()]

        # get largest distance between cameras and ray intersection
        distances = []
        for camera_pose in camera_poses:
            distances.append(np.linalg.norm(self.poi - camera_pose.tf[:3, 3]))
        max_distance = np.max(distances)

        width = max_distance * 1.5
        height = max_distance * 1.5
        depth = max_distance * 1.5

        start_time = time.time()
        self.voxel_grid = VoxelGrid(
            width=width,
            height=height,
            depth=depth,
            voxel_size=voxel_size,
            origin=np.array([self.poi[0] - width/2, self.poi[1] - height/2, self.poi[2] - depth/2]).astype(np.float64),
            color=np.array([0.2, 0.2, 0.2]).astype(np.float64),
            img_width=self.img_width,
            img_height=self.img_height
        )

        self.carve_silhouette(trigger_image, keep_voxels_outside_image=False)

        images = [image for image in self.annotation_images.values() if image.annotation_accepted]
        for image in images:
            self.carve_silhouette(image, keep_voxels_outside_image=True)


    def instanciate_voxel_grid_at_poi_with_prefiltering(self, voxel_size=0.005):

        #TODO maybe add warning if obejcts are further away from the poi than 0.5m

        start_time = time.time()
        camera_poses = [image.camera_pose for image in self.annotation_images.values()]

        # get largest distance between cameras and ray intersection
        distances = []
        for camera_pose in camera_poses:
            distances.append(np.linalg.norm(self.poi - camera_pose.tf[:3, 3]))
        max_distance = np.max(distances)

        width = max_distance * 1.5
        height = max_distance * 1.5
        depth = max_distance * 1.5

        self.voxel_grid = VoxelGrid(
            width=width,
            height=height,
            depth=depth,
            voxel_size=voxel_size,
            origin=np.array([self.poi[0] - width/2, self.poi[1] - height/2, self.poi[2] - depth/2]).astype(np.float64),
            color=np.array([0.2, 0.2, 0.2]).astype(np.float64),
            img_width=self.img_width,
            img_height=self.img_height
        )

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

        start_time = time.time()
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
    
    def next_image_name(self):
        active_image = self.active_image.rgb_path.name
        rgb_imgs = self.annotation_images.keys()
        indx = list(rgb_imgs).index(active_image)
        #if last image return same image and give warning
        if indx == len(rgb_imgs)-1:
            print("Last image reached")
            return active_image
        return list(rgb_imgs)[indx+1]
    

    def write_bop_camera_json(self, output_path, cam_intrinsics_final):
        with open(output_path/"camera.json", 'w') as file:
            json.dump({
                'cx': cam_intrinsics_final.cx,
                'cy': cam_intrinsics_final.cy,
                'fx': cam_intrinsics_final.fx,
                'fy': cam_intrinsics_final.fy,
                'height': cam_intrinsics_final.height,
                'width': cam_intrinsics_final.width,
                'depth_scale': 1.0
            }, file, indent=2) 

    def render_masks_vis(self, cam_intrinsics, cam_poses_world_cords, obj_meshes, obj_ids, obj_poses):
        # scene_cameras = dict()
        cam_intrinsics_final = deepcopy(cam_intrinsics)

        scene_renderer = SceneRenderer(cam_intrinsics_final, obj_meshes, obj_ids, obj_poses)
        masks = []
        for ii, cam_pose_world in enumerate(tqdm(cam_poses_world_cords, total=len(cam_poses_world_cords))):
            cam_pose_world_final = deepcopy(cam_pose_world)


            # fill ii with leading zeros
            img_id = f"{ii+1:06d}"
            image = self.annotation_images[img_id+".png"]

            num_annotation_objs = len(image.annotation_objects)
            annotation_masks = [obj.mask for obj in image.annotation_objects.values()]

            masks_visible = scene_renderer.render_masks(cam_pose_world_final.tf)

            for i in range(len(masks_visible)):
                for j in range(num_annotation_objs):
                    annotation_mask = annotation_masks[j]
                    if annotation_mask is not None:
                        masks_visible[i] = masks_visible[i] & ~annotation_mask
            for i in range(num_annotation_objs):
                if annotation_masks[i] is None:
                    annotation_masks[i] = np.zeros_like(masks_visible[0])
        



            masks_visible += annotation_masks

            masks.append(masks_visible)
        return masks

    def render_masks_all(self, cam_intrinsics, cam_poses_world_cords, obj_meshes, obj_ids, obj_poses):
        cam_intrinsics_final = deepcopy(cam_intrinsics)

        object_renderers = [
            SceneRenderer(cam_intrinsics_final, [obj_mesh], [obj_id], [obj_pose])
            for (obj_mesh, obj_id, obj_pose) in zip(obj_meshes, obj_ids, obj_poses)
        ]

        masks = []
        for ii, cam_pose_world in enumerate(tqdm(cam_poses_world_cords, total=len(cam_poses_world_cords))):
            cam_pose_world_final = deepcopy(cam_pose_world)
            masks_all = [
                r.render_masks(cam_pose_world_final.tf)[0] for r in object_renderers
            ]
            # masks_all = []
            # for r in object_renderers:
            #     mask= r.render_masks(cam_pose_world_final.tf)
            #     for m in mask:
            #         cv2.imshow("mask", m.astype(np.uint8) * 255)
            #         cv2.waitKey(0)
            #         cv2.destroyAllWindows()
            #     masks_all.append(mask[0])
            masks.append(masks_all)
        return masks


    def get_annotation_object_ids(self):
        image = next(iter(self.annotation_images.values()))
        scene_obj_ids = [obj.scene_object_id for obj in image.annotation_objects.values()]
        datatset_obj_ids = [obj.dataset_object_id for obj in image.annotation_objects.values()]
        return scene_obj_ids, datatset_obj_ids


    def get_gt_jsons(self, cam_poses_world_cords, object_poses, object_dataset_ids, masks_all, masks_visible, depth, OBJ_3D_DAT_TO_BOP_ID):

        scene_gts = dict()
        scene_gts_info = dict()
        for ii, cam_pose_world in enumerate(tqdm(cam_poses_world_cords, total=len(cam_poses_world_cords))):
            cam_pose_world_final = deepcopy(cam_pose_world)
            scene_gts[str(ii)] = []
            scene_gts_info[str(ii)] = []
            obj_counter = 0
            for oi, (obj_pose, obj_id) in enumerate(zip(object_poses, object_dataset_ids)):

                obj_pose_world_cords = np.array(obj_pose).reshape((4, 4))
                obj_pose_cam = np.linalg.inv(cam_pose_world_final.tf) @ obj_pose_world_cords

                R_floats = [float(v) for v in obj_pose_cam[:3, :3].reshape(-1)]
                t_floats = [float(v) * 1000 for v in obj_pose_cam[:3, 3].reshape(-1)]  # mm

                mask_all = masks_all[ii][oi]
                mask_visible = masks_visible[ii][oi]

                all_vs, all_us = np.nonzero(mask_all)
                visible_vs, visible_us = np.nonzero(mask_visible)

                if len(visible_us) == 0 or len(visible_vs) == 0:
                    continue

                

                obj_info = {
                        "bbox_obj": get_bbox_from_mask(masks_all[ii][oi]),
                        "bbox_visib": get_bbox_from_mask(masks_visible[ii][oi]),
                        "px_count_all": int(len(all_vs)),
                        "px_count_visib": int(len(visible_vs)),
                        "visib_fract": float(1. * len(visible_vs) / len(all_vs))
                }

                try:
                    obj_info["px_count_valid"] = int((depth[all_vs, all_us] != 0).sum())
                except Exception as e:
                    pass      
                scene_gts[str(ii)].append(
                    {"cam_R_m2c": R_floats, "cam_t_m2c": t_floats, "obj_id": OBJ_3D_DAT_TO_BOP_ID[obj_id]}
                )
                scene_gts_info[str(ii)].append(obj_info)
                obj_counter += 1
        return scene_gts, scene_gts_info

    def get_scene_cameras(self, cam_intrinsics_final, cam_poses_world_cords):
        scene_cameras = dict()
        for ii, cam_pose_world in enumerate(tqdm(cam_poses_world_cords, total=len(cam_poses_world_cords))):


            # scene camera extrinsics in world coordinates and intrinsics
            cam_R_floats = [float(v) for v in cam_pose_world.tf[:3, :3].reshape(-1)]
            cam_t_floats = [float(v) * 1000 for v in cam_pose_world.tf[:3, 3].reshape(-1)]

            # prepare and store scene camera to bop
            K = np.array([[cam_intrinsics_final.fx, 0, cam_intrinsics_final.cx],
                            [0, cam_intrinsics_final.fy, cam_intrinsics_final.cy],
                            [0, 0, 1.0]])

            scene_cameras[str(ii)] = {"cam_K": K.reshape(-1).tolist(), "depth_scale": 1.0,
                                                "cam_R_w2c": cam_R_floats, "cam_t_w2c": cam_t_floats}
        return scene_cameras
    
    def write_bop_files(
            self,
            scene_path_bop, 
            cam_poses_world_cords, 
            rgbs, 
            depths, 
            masks_all, 
            masks_visible, 
            objects, 
            scene_gts, 
            scene_gts_info, 
            scene_cameras,
            only_masks):
        if not only_masks:
            os.makedirs(os.path.join(scene_path_bop, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(scene_path_bop, "depth"), exist_ok=True)
        os.makedirs(scene_path_bop, exist_ok=True)

        os.makedirs(os.path.join(scene_path_bop, "mask"), exist_ok=True)
        os.makedirs(os.path.join(scene_path_bop, "mask_visib"), exist_ok=True)

        for ii, cam_pose_world in enumerate(tqdm(cam_poses_world_cords, total=len(cam_poses_world_cords))):
            img_id = f"{ii:06d}"
            if not only_masks:
                cv2.imwrite(os.path.join(scene_path_bop, f"rgb/{img_id}.png"), rgbs[ii])
                cv2.imwrite(os.path.join(scene_path_bop, f"depth/{img_id}.png"), depths[ii])

            obj_counter = 0
            for oi, obj in enumerate(objects):
                obj_id = f"{obj_counter:06d}"
                cv2.imwrite(os.path.join(scene_path_bop, f"mask_visib/{img_id}_{obj_id}.png"),
                    255 * masks_visible[ii][oi].astype(np.uint8))
                cv2.imwrite(os.path.join(scene_path_bop, f"mask/{img_id}_{obj_id}.png"),
                    255 * masks_all[ii][oi].astype(np.uint8))
                obj_counter += 1

        with open(f"{scene_path_bop}/scene_gt.json", 'w') as file:
            json.dump(scene_gts, file, indent=2)
        with open(f"{scene_path_bop}/scene_gt_info.json", 'w') as file:
            json.dump(scene_gts_info, file, indent=2)
        with open(f"{scene_path_bop}/scene_camera.json", 'w') as file:
            json.dump(scene_cameras, file, indent=2)

    def write_to_bop(self, path, mode, experiment=None):
        import os

        # find which scene this is
        scene_ids = self.scene_reader.get_scene_ids()

        object_lib = self.scene_reader.get_object_library()
        for si, scene_id in enumerate(scene_ids):
            if scene_id == self.scene_id:
                break
        
        #annotation_ids
        annotation_obj_scene_ids, annotation_obj_dataset_ids = self.get_annotation_object_ids()
        #scene_id : dataset_id
        OBJ_3D_DAT_TO_BOP_ID = {
            obj_id: int(obj.mesh.file.split('/')[-1][4:-4])
            for obj_id, obj in object_lib.items()
        }
        for dataset_obj_id in  annotation_obj_dataset_ids:
            OBJ_3D_DAT_TO_BOP_ID[dataset_obj_id] = dataset_obj_id


        # write_scene_to_bop(path, si, self.scene_id, self.scene_reader, OBJ_3D_DAT_TO_BOP_ID, "train")
        
        #get_object_meshes
        annotation_ids = [obj.scene_object_id for obj in next(iter(self.annotation_images.values())).annotation_objects.values()]

        meshes, poses, ids = self.voxel_grid.convert_voxel_grid_to_mesh(ids=annotation_ids)
        dataset_ids = []
        for id in ids:
            id_indx = annotation_obj_scene_ids.index(id)
            dataset_ids.append(annotation_obj_dataset_ids[id_indx])

        scene_id = self.scene_id
        scene_file_reader = self.scene_reader
        cam_poses_world_cords = scene_file_reader.get_camera_poses(scene_id)
        cam_intrinsics = scene_file_reader.get_camera_info_scene(scene_id)
        object_poses = scene_file_reader.get_object_poses(scene_id)

        #TODO make sure nothing is upside down during annotation
        cam_intrinsics_final = deepcopy(cam_intrinsics)

        if experiment == None:
            scene_path_bop = path/f"{mode}/{(si):06d}"
        else:
            scene_path_bop = path/f"{mode}/{(si):06d}_{(experiment):03d}"
        os.makedirs(scene_path_bop, exist_ok=True)
        self.write_bop_camera_json(scene_path_bop, deepcopy(cam_intrinsics_final))
            
        # save trimesh as ply into dataset_path/models

        mesh_folder = path/"models"/f"{(si):06d}"
        if experiment != None:
            ply_name = f"obj_{int(id):06d}_{(experiment):03d}.ply"
        else:
            ply_name = f"obj_{int(id):06d}.ply"
        os.makedirs(mesh_folder, exist_ok=True)
        for mesh, id in zip(meshes, dataset_ids):
            mesh.export(file_obj=mesh_folder/ply_name)


        obj_meshes = []
        obj_poses = []
        obj_ids = []
        obj_dataset_ids = []
        for oi, (obj, obj_pose) in enumerate(object_poses):
            obj_meshes.append(obj.mesh.as_trimesh())
            obj_poses.append(obj_pose)
            obj_ids.append(oi)
            obj_dataset_ids.append(obj.id)


        vis_masks = self.render_masks_vis(deepcopy(cam_intrinsics), deepcopy(cam_poses_world_cords), deepcopy(obj_meshes), deepcopy(obj_ids), deepcopy(obj_poses))

        all_obj_meshes = obj_meshes + meshes
        all_obj_ids = obj_ids + annotation_obj_scene_ids
        all_obj_dataset_ids = obj_dataset_ids + annotation_obj_dataset_ids

        all_obj_poses = obj_poses + poses

        full_masks = self.render_masks_all(deepcopy(cam_intrinsics), deepcopy(cam_poses_world_cords), deepcopy(all_obj_meshes), deepcopy(all_obj_ids), deepcopy(all_obj_poses))

        rgb_paths = scene_file_reader.get_images_rgb_path(scene_id)
        depth_paths = scene_file_reader.get_images_depth_path(scene_id)

        rgbs = [cv2.imread(rgb_path) for rgb_path in rgb_paths]
        depths = [cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) for depth_path in depth_paths]

        scene_gts, scene_gts_info = self.get_gt_jsons(
            deepcopy(cam_poses_world_cords),
            deepcopy(all_obj_poses), 
            deepcopy(all_obj_dataset_ids),
            deepcopy(full_masks), 
            deepcopy(vis_masks), 
            deepcopy(depths), 
            deepcopy(OBJ_3D_DAT_TO_BOP_ID))

        scene_cameras = self.get_scene_cameras(deepcopy(cam_intrinsics_final), deepcopy(cam_poses_world_cords))
        if experiment == None:
            only_masks = False
        else:
            only_masks = True
        self.write_bop_files(
            deepcopy(scene_path_bop), 
            deepcopy(cam_poses_world_cords), 
            deepcopy(rgbs), 
            deepcopy(depths), 
            deepcopy(full_masks), 
            deepcopy(vis_masks), 
            deepcopy(all_obj_meshes), 
            deepcopy(scene_gts), 
            deepcopy(scene_gts_info), 
            deepcopy(scene_cameras),
            only_masks=only_masks)

        #write scene

        # intrinsics
        # cameras poses
        # object meshes
        # obejct poses



        # enumerate_find index_write to bop -> done

        

        # substract anno masks for occlusion
        # estimate mask from voxelgrid
        # calc centerpoint of voxelgrid for translation
        # np. eye for rotation
        # append poses
        # convert to coco




