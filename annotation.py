from dataclasses import dataclass
from utils.v4r import SceneFileReader
import numpy as np
import cv2
from pathlib import Path
from copy import deepcopy
import utils.vis_masks as vis_masks
import open3d as o3d
import mcubes

@dataclass
class AnnotationObject:
    prompts: list
    prompts_label: list
    mask: np.ndarray
    logit: np.ndarray
    label: str
class AnnotationImage:
    def __init__(self, rgb_path, camera_pose, silhouette=None):
        self.rgb_path = rgb_path
        self.camera_pose = camera_pose
        self.annotation_objects = {}
        self.silhouette = silhouette
        self.active_object = None
        self.annotation_accepted = False

    def generate_visualization(self):
        image = cv2.imread(self.rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        overlay = image.copy()

        light_blue = (73, 116, 130)
        light_red = (155, 82, 93)

        for prompt_obj in self.annotation_objects.values():
            if prompt_obj.mask is None:
                continue
            color = light_red
            if prompt_obj == self.active_object:
                color = light_blue
            mask = prompt_obj.mask.astype(np.uint8)

            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color

            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)

        for prompt_obj in self.annotation_objects.values():       
            if prompt_obj.mask is None:
                continue     
            color = (255, 0, 0)
            if prompt_obj == self.active_object:
                color = (0, 0, 255)

            contours, _ = cv2.findContours(prompt_obj.mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        for (x, y), label in zip(self.active_object.prompts, self.active_object.prompts_label):
            dot_color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.rectangle(overlay, (x-2, y-2), (x+2, y+2), dot_color, -1)

        return overlay
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

    def load_images(self):
        image_paths = self.scene_reader.get_images_rgb_path(self.scene_id)
        camera_poses = self.scene_reader.get_camera_poses(self.scene_id)
        rigit_silhouette = self.get_rigit_silhouette()

        #perform smart camera pose ordering
        reordering = self.max_distance_camera_reorder(camera_poses)
        # camera_poses = [camera_poses[i] for i in reordering]
        # image_paths = [image_paths[i] for i in reordering]
        # rigit_silhouette = rigit_silhouette[reordering]
        # for i, image_path in enumerate(image_paths):
        #     image_path = Path(image_path)
        #     self.annotation_images[image_path.name] = AnnotationImage(image_path, camera_poses[i], rigit_silhouette[i])
        for i in reordering:
            image_path = Path(image_paths[i])
            self.annotation_images[image_path.name] = AnnotationImage(image_path, camera_poses[i], rigit_silhouette[i])
    
    def get_rigit_silhouette(self):
        # read in all scene masks

        # mask count
        mask_path = Path(self.scene_reader.get_images_mask_path(self.scene_id))

        pose_count = len(self.scene_reader.get_camera_poses(self.scene_id))
        object_count = len(self.scene_reader.get_object_poses(self.scene_id))
        width = self.scene_reader.get_camera_info_scene(self.scene_id).width
        height = self.scene_reader.get_camera_info_scene(self.scene_id).height
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
        
    def max_distance_camera_reorder(self, poses, k=7):
        # if k is None:
        #     k = int(len(poses)/2)
        points = np.array([pose.translation() for pose in poses])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        downsampled_points = np.asarray(pcd.farthest_point_down_sample(k).points)
        #sort the downsampled list by z value decreasing
        # downsampled_points = downsampled_points[np.argsort(downsampled_points[:, 2])[::-1]]

        # return reordering of original list so that the downsampled_points are first
        
        # find indices of poses in original list
        indices = [np.where(points == pose)[0][0] for pose in downsampled_points]
        full_odering = indices + [i for i in range(len(points)) if i not in indices]
        return full_odering
    
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
            new_voxel = o3d.geometry.Voxel(pos, [0, 0, 1])
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
        mask = image.silhouette.astype(np.uint8)
        for obj in image.annotation_objects.values():
            if obj.mask is None:
                continue
            mask += obj.mask.astype(np.uint8)
        mask[mask > 0] = 1

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

class VoxelGrid:
    def __init__(self, width, height, depth, voxel_size, origin, color):
        self.width = width
        self.height = height
        self.depth = depth
        self.voxel_size = voxel_size
        self.origin = origin
        self.color = color
        self.o3d_grid = o3d.geometry.VoxelGrid.create_dense(
            origin=self.origin,
            color=self.color,
            voxel_size=self.voxel_size,
            width=self.width,
            height=self.height,
            depth=self.depth
        )

    def get_voxel_grid_top_down_view(self, z=1):
        poi = [self.origin[0] + self.width/2, self.origin[1] + self.height/2, self.origin[2] + self.depth/2]
        renderer = o3d.visualization.rendering.OffscreenRenderer(500, 500)

        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA, does not replace the mesh color
        mtl.shader = "defaultUnlit"

        renderer.scene.add_geometry("grid", self.o3d_grid, mtl)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(500, 500, 250, 250, 250, 250)
        #extrensics: translation 2 meters above self.poi in z. Looking down
        
        pose = np.array([
            [1, 0, 0, poi[0]],
            [0, 1, 0, poi[1]],
            [0, 0, 1, poi[2]-z],
            [0, 0, 0, 1]
        ])

        extrinsics = np.linalg.inv(pose)

        renderer.setup_camera(intrinsics, extrinsics)
        img = np.asarray(renderer.render_to_image())
        return img

    def visualize_colored_meshes(self, meshes):
        def get_random_color():
            return list(np.random.choice(range(256), size=3) / 255.0)
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        for mesh in meshes:
            color = get_random_color()
            mesh.paint_uniform_color(color)  
            
            vis.add_geometry(mesh)
        vis.add_geometry(self.o3d_grid)
        vis.run()
        vis.destroy_window()

    def convert_voxel_grid_to_mesh(self):

        voxel_grid = np.zeros((int(self.width/self.voxel_size), int(self.height/self.voxel_size), int(self.depth/self.voxel_size)))
        print(voxel_grid.shape)
        for voxel in self.o3d_grid.get_voxels():
            voxel_grid[voxel.grid_index[0], voxel.grid_index[1], voxel.grid_index[2]] = 15
        
        vertices, triangles = mcubes.marching_cubes(voxel_grid, 0)
        print(len(vertices), len(triangles))
        print(vertices[0], triangles[0])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        o3d.visualization.draw_geometries([mesh])
        print(f"mesh center: mesh.get_center()")

        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        # Get unique cluster labels and corresponding triangle counts
        unique_labels = np.unique(triangle_clusters)
        print(unique_labels)

        # Create separate meshes for each cluster
        separate_meshes = []
        for label in unique_labels:
            # Filter triangles belonging to the current cluster
            cluster_triangles = np.where(triangle_clusters == label)[0]
            if len(cluster_triangles) < 100:
                continue
            mesh_cluster = o3d.geometry.TriangleMesh(
                vertices=mesh.vertices,
                triangles=o3d.utility.Vector3iVector(
                    np.asarray(mesh.triangles)[cluster_triangles]
                ),
            )
            separate_meshes.append(mesh_cluster)
        self.visualize_colored_meshes(separate_meshes)

class AnnotationDataset:
    def __init__(self, dataset_path, config="config.cfg"):
        self.dataset_path = dataset_path
        self.scene_reader = SceneFileReader.create(dataset_path/config)
        self.annotation_scenes = {}
        self.active_scene = None
        self.load_scenes()

    def load_scenes(self):
        scene_ids = self.scene_reader.get_scene_ids()
        for scene_id in scene_ids:
            camera = self.scene_reader.get_camera_info_scene(scene_id)
            camera_intrinsics = camera.as_numpy3x3()
            self.annotation_scenes[scene_id] = AnnotationScene(scene_id, self.scene_reader, camera_intrinsics, camera.width, camera.height)

    def get_scene_ids(self):
        return self.annotation_scenes.keys()

from annotation import AnnotationDataset
from pathlib import Path

if __name__ == "__main__":
    dataset_path = Path("/home/daviddylan/Interdisciplinary/depth-estimation-of-transparent-objects/")

    dataset = AnnotationDataset(dataset_path, config="config.cfg")
    dataset.annotation_scenes['j_002'].load_images()
    print(dataset.annotation_scenes['j_002'].has_correct_number_of_masks())
