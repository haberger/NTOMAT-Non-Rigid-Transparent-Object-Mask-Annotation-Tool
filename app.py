import gradio as gr
import argparse
from pathlib import Path
import time
import utils.vis_masks as vis_masks
from utils.v4r import SceneFileReader
import cv2
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

DATASET_PATH = None
image_height = None
image_width = None
predictor = None
prompter = None
scene_reader = None


js_events = """
<script>
function clickHandler(e) {
    var image_input = document.getElementById("prompting_image").querySelector('img');
    if (!image_input) return; // Make sure the image element exists

    var imgWidth = image_input.width;
    var imgHeight = image_input.height;

    var rect = image_input.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;

    if (x < 0 || x > imgWidth || y < 0 || y > imgHeight) {
        return;
    }

    var button_label = e.button == 2 ? "right" : "left";
    var js_parser = document.getElementById("js_parser").querySelector('textarea');
    js_parser.value = `${Math.round(x)} ${Math.round(y)} ${button_label} ${imgWidth} ${imgHeight}`;    
    js_parser.dispatchEvent(new Event('input', { bubbles: true }));
}
document.addEventListener('mousedown', clickHandler, false);
document.addEventListener('contextmenu', function(e) {
    e.preventDefault(); // Prevent the context menu from appearing
}, false);
</script>
"""

@dataclass
class PromptObject:
    prompts: list
    prompts_label: list
    mask: np.ndarray
    logit: np.ndarray
    label: str

@dataclass
class Prompter:
    prompt_objects: list
    image: Path
    scene_id: str
    active_object: int

    def generate_visualization(self):
        image = cv2.imread(str(self.image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        overlay = image.copy()

        light_blue = (73, 116, 130)
        light_red = (155, 82, 93)

        for i, prompt_obj in enumerate(self.prompt_objects):
            color = light_red
            if i == self.active_object:
                color = light_blue

            mask = prompt_obj.mask.astype(np.uint8)

            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color

            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)

        for i, prompt_obj in enumerate(self.prompt_objects):
            color = (255, 0, 0)
            if i == self.active_object:
                color = (0, 0, 255)

            contours, _ = cv2.findContours(prompt_obj.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        active_prompt_obj = self.prompt_objects[self.active_object]
        for (x, y), label in zip(active_prompt_obj.prompts, active_prompt_obj.prompts_label):
            dot_color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(overlay, (x, y), 5, dot_color, -1)

        return overlay
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def load_scene(scene_id, img_selection):
    global scene_reader
    global prompter

    yield f"Loading Scene {scene_id}:", None, img_selection

    # Check if masks directory exists
    masks_dir = Path(scene_reader.root_dir) / scene_reader.scenes_dir / scene_id / scene_reader.mask_dir

    if not masks_dir.exists(): 
        yield f"Loading Scene {scene_id}: Generating missing masks", None, img_selection
        vis_masks.create_masks(scene_reader, scene_id)
    else:
        expected_mask_count = len(scene_reader.get_object_poses(scene_id)) * len(scene_reader.get_camera_poses(scene_id))
        if expected_mask_count != len(list(masks_dir.iterdir())):
            gr.Warning(f"Missing masks for scene {scene_id} generating new masks", duration=3)
            yield f"Loading Scene {scene_id}: Generating missing masks", None, img_selection
            vis_masks.create_masks(scene_reader, scene_id)

    # Load first image of scene as default np array from path
    rgb_imgs_path = scene_reader.get_images_rgb_path(scene_id)
    rgb_imgs_path = [Path(p) for p in rgb_imgs_path]

    img_selection = gr.Dropdown(
        value = rgb_imgs_path[0].name, 
        choices = [f.name for f in rgb_imgs_path],
        label = "Select an Image",
        visible=True
    )

    prompter = Prompter([], None, scene_id, None)

    yield f"Loaded Scene {scene_id}!", img_selection

def click_image(image, evt: gr.SelectData):
    return

def js_trigger(input_data, image):
    global prompter
    data = dict(zip(["x", "y", "button", "imgWidth", "imgHeight"], input_data.split()))
    print(data)
    #TODO factor in image size -> if scaled, need to scale back to original size

    print("Image Clicked")

    # if first predict full
    #take best mask and visualize
    #take best logit and use for further predictions

    input_point = [[int(data['x']), int(data['y'])]]
    input_label = [0] if data['button'] == 'right' else [1]

    if prompter.active_object is None:
        masks, scores, logits = predictor.predict(
            point_coords=np.array(input_point),
            point_labels=np.array(input_label),
            multimask_output=True,
        )
        
        best_mask = masks[np.argmax(scores), :, :]
        best_logit = logits[np.argmax(scores), :, :]
        prompt_object = PromptObject(input_point, input_label, best_mask, best_logit, "test")
        prompter.prompt_objects.append(prompt_object)
        prompter.active_object = 0
    else:
        prompt_object = prompter.prompt_objects[prompter.active_object]

        # input_point = np.array([[500, 375], [1125, 625]])
        # input_label = np.array([1, 1])

        prompt_object.prompts.append([int(data['x']), int(data['y'])])
        prompt_object.prompts_label.append(int(input_label[0]))

        print(prompt_object.prompts)
        print(prompt_object.prompts_label)
        print(prompt_object.logit.shape)
        mask, score, logit = predictor.predict(
            point_coords=np.array(prompt_object.prompts),
            point_labels=np.array(prompt_object.prompts_label),
            mask_input = prompt_object.logit[None, :, :],
            multimask_output=False,
        )

        prompt_object.mask = mask[0,:,:]
        prompt_object.logit = logit[0,:,:]
        prompter.prompt_objects[prompter.active_object] = prompt_object

    print(prompt_object.mask.shape)

    image = prompter.generate_visualization()
    return  -1, image

def change_image(img_selection):
    print("Image Changed")
    global prompter
    global scene_reader
    global predictor
    if prompter.image is not None:
        #TODO save prompter
        pass
    
    prompter.image = Path(scene_reader.root_dir)/scene_reader.scenes_dir/prompter.scene_id/scene_reader.rgb_dir/img_selection
    img = cv2.cvtColor(cv2.imread(str(prompter.image)), cv2.COLOR_BGR2RGB)
    predictor.set_image(img)
    return img


def main(dataset_path):
    global DATASET_PATH
    global prompter
    global predictor
    global scene_reader
    DATASET_PATH = dataset_path

    sam_checkpoint = "model_checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # Get list of folders in dataset_path
    scene_reader = SceneFileReader.create(dataset_path / 'config.cfg')
    scene_folders = sorted([f.stem for f in (dataset_path / 'scenes').iterdir() if f.is_dir()])

    # predictor.set_image(im)

    with gr.Blocks(head=js_events) as demo:
        status_md = gr.Markdown(f"Select a Folder from Dataset {dataset_path}")
        # prompting_state = gr.State()
        # Create a dropdown to select a scene
        with gr.Row():
            folder_selection = gr.Dropdown(
                choices = scene_folders,
                label = "Select a Scene"
            )
            img_selection = gr.Dropdown(
                choices = [],
                label = "Select an Image",
                visible=False
            )
        js_parser = gr.Textbox(label="js_parser", elem_id="js_parser", visible=False)
        # #IDEA maybe hide until scene is selected 
        prompting_image = gr.Image(label="Upload Image", elem_id="prompting_image", elem_classes="images") 
        folder_selection.change(load_scene, inputs=[folder_selection, img_selection], outputs=[status_md, img_selection])
        img_selection.change(change_image, inputs=[img_selection], outputs=[prompting_image])
        prompting_image.select(click_image, [prompting_image])
        js_parser.input(js_trigger, [js_parser, prompting_image], [js_parser, prompting_image])
    demo.queue()
    demo.launch()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NTOMAT')
    parser.add_argument(  
        '-d',
        dest='dataset_path',
        default = '../depth-estimation-of-transparent-objects',
        help='path_to_3D-DAT Dataset')

    args = parser.parse_args()

    main(Path(args.dataset_path))