import cv2
import numpy as np
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

@dataclass
class AnnotationObject:
    prompts: list
    prompts_label: list
    mask: np.ndarray
    logit: np.ndarray
    label: str
    dataset_object_id: str
    scene_object_id: str
    

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

    def generate_auto_prompts(self, scene, predictor):
        '''
        takes an annotation object and generates prompts for it
        '''

        voxelgrid = scene.voxel_grid
        voxelgrid_segmap = voxelgrid.project_voxelgrid(scene.img_width, scene.img_height, scene.camera_intrinsics, self.camera_pose, voxelgrid.o3d_grid_id)
        voxelgrid_segmap = voxelgrid_segmap[:,:,0]


        print("generating prompts")

        for obj in self.annotation_objects.values():
            if obj.mask is not None:
                print("mask not none")
                continue
            mask = np.zeros_like(voxelgrid_segmap)
            mask[voxelgrid_segmap == obj.scene_object_id] = 255
            
            #3. generate prompts
            prompt_points = self.get_prompt_points_from_mask(mask, debug_visualization=True)
            if prompt_points is not None:
                for point in prompt_points:
                    self.active_object = obj
                    self.add_prompt([[point[1], point[0]]], [1], predictor)

            for scene_object_id in scene.scene_object_ids:
                if scene_object_id != obj.scene_object_id:
                    mask = np.zeros_like(voxelgrid_segmap)
                    mask[voxelgrid_segmap == scene_object_id] = 255
                    if np.sum(mask) == 0:
                        continue

                    prompt_points = self.get_prompt_points_from_mask(mask, debug_visualization=True, just_one_point=True)
                    if prompt_points is not None:
                        for point in prompt_points:
                            self.active_object = obj
                            self.add_prompt([[point[1], point[0]]], [0], predictor)

    def erode_with_minimum_points(self, mask, kernel, iterations, min_points_ratio):
        initial_size = np.count_nonzero(mask)
        min_points = int(min_points_ratio * initial_size)

        for _ in range(iterations):
            eroded_mask = cv2.erode(mask, kernel)
            if np.count_nonzero(eroded_mask) < min_points:
                return mask  # Return the original mask before erosion
            mask = eroded_mask

        return mask

    def get_prompt_points_from_mask(self, mask, debug_visualization=True, just_one_point=False):

        # for every connected component erode it until it has 10% of its original size
        all_points = []
        skeleton = np.zeros_like(mask)

        if just_one_point:
            segment_points = np.argwhere(mask > 0)
            segment_points = segment_points.reshape(-1, 2)
            initial_size = len(segment_points)
            segment_mask = mask
            segment_mask = self.erode_with_minimum_points(
                segment_mask, np.ones((3, 3), np.uint8), 1000, 0.1
            )
            segment_points = np.argwhere(segment_mask > 0)
            segment_points = segment_points.reshape(-1, 2)
            skeleton += segment_mask
            if len(segment_points) > 0:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, _, centers = cv2.kmeans(segment_points.astype(np.float32), 1, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
                if len(centers[0]) == 2:
                    all_points.extend(centers)
                else:
                    print(f"Warning: Could not find 2 points for segment for single prompt {centers}")
        else:
            num_labels, labels = cv2.connectedComponents(mask)
            for label in range(1, num_labels):  # Start at 1 to skip the background label (0)
                segment_mask = (labels == label).astype(np.uint8)
                segment_points = np.argwhere(segment_mask > 0)
                segment_points = segment_points.reshape(-1, 2)
                initial_size = len(segment_points)
                if initial_size < 10:
                    continue

                points = np.unique(labels, return_counts=True)[1]
                num_points_per_segment = np.maximum(np.ceil(points / 3000), 1).astype(np.int32)

                if len(segment_points) > 0:
                    segment_mask = self.erode_with_minimum_points(
                        segment_mask, np.ones((3, 3), np.uint8), 1000, 0.1
                    )
                    segment_points = np.argwhere(segment_mask > 0)
                    segment_points = segment_points.reshape(-1, 2)
                    skeleton += segment_mask
                    if len(segment_points) > num_points_per_segment[label]:
                        # Perform k-means
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                        _, _, centers = cv2.kmeans(segment_points.astype(np.float32), num_points_per_segment[label], None, criteria, 10, cv2.KMEANS_PP_CENTERS)

                        # Snap centers to nearest skeleton points (corrected)
                        distances = cdist(centers, segment_points)
                        closest_point_indices = np.argmin(distances, axis=1)
                        snapped_centers = segment_points[closest_point_indices]
                        for center in snapped_centers:
                            if len(center) == 2:
                                all_points.append(center)
                            else:
                                print(f"Warning: Could not find 2 points for segment {label} for multiple prompts {center}")
        centers = np.array(all_points, dtype=np.int32)
            
        # if debug_visualization:
        #     print("WTF")
        #     plt.figure(figsize=(10,10))
        #     plt.imshow(mask)
        #     plt.imshow(skeleton, cmap='gray', alpha=0.5)
        #     plt.scatter(centers[:, 1], centers[:, 0], c='r', s=10)
        #     plt.axis('off')
        #     plt.show()
        #     #plt savefig
        #     # plt.savefig('skeleton_img.png')
        if len(centers) == 0:
            return None

        return centers

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
    
    def get_complete_segmap(self):
        segmap = self.segmap
        for obj in self.annotation_objects.values():
            if obj.mask is not None:
                segmap[obj.mask > 0] = obj.scene_object_id
        return segmap

if __name__ == "__main__":
    image = cv2.imread('test_sceletonization.png')
    #only use 1st channel
    image = image[:,:,0]
    print(image.shape)
    AI = AnnotationImage(None, None)
    AI.get_prompt_points_from_mask(image, debug_visualization=True)


