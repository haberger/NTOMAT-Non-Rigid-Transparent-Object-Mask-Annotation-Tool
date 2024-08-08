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

        #1. generate mask from voxelgrid
        # -> get new segmentation map from voxelgrid (use offscreen rednerer project to pose)
        voxelgrid = scene.voxel_grid
        voxelgrid_segmap = voxelgrid.project_voxelgrid(scene.img_width, scene.img_height, scene.camera_intrinsics, self.camera_pose, voxelgrid.o3d_grid_id)
        voxelgrid_segmap = voxelgrid_segmap[:,:,0]
        #TODO add annotation_objects to scene_ids-> assign scene ids

        #2. for every object in annotation object

        print("generating prompts")

        for obj in self.annotation_objects.values():
            if obj.mask is not None:
                print("mask not none")
                continue
            mask = np.zeros_like(voxelgrid_segmap)
            mask[voxelgrid_segmap == obj.scene_object_id] = 255

            #show mask
            cv2.imshow('mask', mask.astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            #3. generate prompts
            prompt_points = self.get_prompt_points_from_mask(mask)

            # visualize prompt point on rgb image
            img = cv2.imread(self.rgb_path)
            for point in prompt_points:
                cv2.circle(img, (point[1], point[0]), 3, (0, 255, 0), -1)
            cv2.imshow('prompt_points', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            for point in prompt_points:
                self.active_object = obj
                self.add_prompt([[point[1], point[0]]], [1], predictor)

            for scene_object_id in scene.scene_object_ids:
                if scene_object_id != obj.scene_object_id:
                    mask = np.zeros_like(voxelgrid_segmap)
                    mask[voxelgrid_segmap == scene_object_id] = 255

                    prompt_points = self.get_prompt_points_from_mask(mask, debug_visualization=True)
                    for point in prompt_points:
                        self.active_object = obj
                        self.add_prompt([[point[1], point[0]]], [0], predictor)
        print("prompts generated")
        # -> get mask from segmentation map
        # -> generate positive prompts using get_prompt_points_from_mask
        # -> generate negative prompts using get_prompt_points_from_mask for all other objects

        # 3. for each prompt
        # -> add prompt to annotation object using add_prompt

        #TODO on write update obejcts.library
        img = self.generate_visualization()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('auto_prompts', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass

    def get_prompt_points_from_mask(self, mask, debug_visualization=False):
        # show mask
        print(mask.dtype)
        print(type(mask))
        print(np.min(mask))
        print(np.max(mask))
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        skeleton = cv2.ximgproc.thinning(mask)

        #show skeleton
        cv2.imshow('skeleton', skeleton.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
            print(f"""Segment {label} has {len(segment_points)} points""")
            print(num_points_per_segment[label])
            print(segment_points.shape)

            if len(segment_points) > num_points_per_segment[label]:
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
            print(f"""Segment {label} has {len(all_points)} points clusters""")

        centers = np.array(all_points, dtype=np.int32)
        print(f"all points: {all_points}")
        print(f"centers: {centers}")
        print(centers.shape)

        if debug_visualization:
            plt.figure(figsize=(10,10))
            plt.imshow(mask)
            plt.imshow(skeleton, cmap='gray', alpha=0.5)
            plt.scatter(centers[:, 1], centers[:, 0], c='r', s=100)
            plt.axis('off')
            plt.show()
            #plt savefig
            plt.savefig('skeleton_img.png')

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


