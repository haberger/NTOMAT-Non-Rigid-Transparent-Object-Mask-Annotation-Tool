import cv2
import numpy as np
from dataclasses import dataclass
from scipy.spatial.distance import cdist

@dataclass
class AnnotationObject:
    prompts: list
    prompts_label: list
    mask: np.ndarray
    logit: np.ndarray
    label: str

class AnnotationImage:
    def __init__(self, rgb_path, camera_pose, rigid_segmap=None):
        self.rgb_path = rgb_path
        self.camera_pose = camera_pose
        self.annotation_objects = {}
        self.segmap= rigid_segmap
        self.active_object = None
        self.annotation_accepted = False

    def add_prompt(self, input_point, input_label, predictor):
        if self.active_object.mask is None:
            masks, scores, logits = predictor.predict(
                point_coords=np.array(input_point),
                point_labels=np.array(input_label),
                multimask_output=True,
            )

            self.active_object.prompts.append(input_point[0])
            self.active_object.prompts_label.append(int(input_label[0]))
            self.active_object.mask = masks[np.argmax(scores), :, :]
            self.active_object.logit = logits[np.argmax(scores), :, :]

        else:

            self.active_object.prompts.append(input_point[0])
            self.active_object.prompts_label.append(int(input_label[0]))

            mask, score, logit = predictor.predict(
                point_coords=np.array(self.active_object.prompts),
                point_labels=np.array(self.active_object.prompts_label),
                mask_input = self.active_object.logit[None, :, :],
                multimask_output=False,
            )

            self.active_object.mask = mask[0,:,:]
            self.active_object.logit = logit[0,:,:]

        #write mask with unique id using cv2
        cv2.imwrite('mask_tube.png', self.active_object.mask.astype(np.uint8)*255)

    def generate_auto_prompts(self, scene):
        '''
        takes an annotation object and generates prompts for it
        '''

        #1. generate mask from voxelgrid
        # -> get new segmentation map from voxelgrid (use offscreen rednerer project to pose)
        voxelgrid = scene.voxel_grid
        voxelgrid.project_voxelgrid(scene.img_width, scene.img_height, scene.camera_intrinsics, self.camera_pose, voxelgrid.o3d_grid_id)

        #TODO add annotation_objects to scene_ids-> assign scene ids

        #2. for every object in annotation object
        # -> get mask from segmentation map
        # -> generate positive prompts using get_prompt_points_from_mask
        # -> generate negative prompts using get_prompt_points_from_mask for all other objects

        # 3. for each prompt
        # -> add prompt to annotation object using add_prompt

        pass

    def get_prompt_points_from_mask(self, mask):
        skeleton = cv2.ximgproc.thinning(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

        num_labels, labels = cv2.connectedComponents(skeleton)
        num_points_per_segment = 1

        #get num of points in each segment
        points = np.unique(labels, return_counts=True)[1]
        num_points_per_segment = np.maximum(np.ceil(points / 300), 1).astype(np.int32)

        all_points = []  
        for label in range(1, num_labels):  # Start at 1 to skip the background label (0)
            segment_mask = (labels == label).astype(np.uint8)
            segment_points = np.argwhere(segment_mask > 0)
            segment_points = segment_points.reshape(-1, 2)

            if len(segment_points) >= num_points_per_segment[label]:
                # Perform k-means
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, _, centers = cv2.kmeans(segment_points.astype(np.float32), num_points_per_segment[label], None, criteria, 10, cv2.KMEANS_PP_CENTERS)

                # Snap centers to nearest skeleton points (corrected)
                distances = cdist(centers, segment_points)
                closest_point_indices = np.argmin(distances, axis=1)  # Find closest for EACH center
                snapped_centers = segment_points[closest_point_indices]
                all_points.extend(snapped_centers)

            elif len(segment_points) > 0:
                # If too few points for k-means, still include one
                center_index = np.random.choice(len(segment_points)) 
                all_points.append(segment_points[center_index])

        centers = np.array(all_points, dtype=np.int32)

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

