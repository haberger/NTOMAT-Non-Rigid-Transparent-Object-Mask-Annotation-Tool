import cv2
import numpy as np
from dataclasses import dataclass

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

