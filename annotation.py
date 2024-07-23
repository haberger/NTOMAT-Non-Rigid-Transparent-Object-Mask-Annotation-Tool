from dataclasses import dataclass
from utils.v4r import SceneFileReader
import numpy as np
import cv2
from pathlib import Path
import utils.vis_masks as vis_masks
import open3d as o3d

@dataclass
class AnnotationObject:
    prompts: list
    prompts_label: list
    mask: np.ndarray
    logit: np.ndarray
    label: str

class AnnotationImage:
    def __init__(self, rgb_path, camera_extrinsics, loaded_mask=None):
        self.rgb_path = rgb_path
        self.camera_extrinsics = camera_extrinsics
        self.annotation_objects = {}
        self.loaded_mask = None
        self.active_object = None

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

    def load_images(self):
        image_paths = self.scene_reader.get_images_rgb_path(self.scene_id)
        camera_poses = self.scene_reader.get_camera_poses(self.scene_id)

        #perform smart camera pose ordering
        reordering = self.max_distance_camera_reorder(camera_poses)
        camera_poses = [camera_poses[i] for i in reordering]
        image_paths = [image_paths[i] for i in reordering]

        for i, image_path in enumerate(image_paths):
            image_path = Path(image_path)
            self.annotation_images[image_path.name] = AnnotationImage(image_path, camera_poses[i])
    
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
        
    def max_distance_camera_reorder(self, poses, k=5):
        points = np.array([pose.translation() for pose in poses])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        downsampled_points = np.asarray(pcd.farthest_point_down_sample(k).points)
        #sort the downsampled list by z value decreasing
        downsampled_points = downsampled_points[np.argsort(downsampled_points[:, 2])[::-1]]

        # return reordering of original list so that the downsampled_points are first
        
        # find indices of poses in original list
        indices = [np.where(points == pose)[0][0] for pose in downsampled_points]
        full_odering = indices + [i for i in range(len(points)) if i not in indices]
        return full_odering

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
