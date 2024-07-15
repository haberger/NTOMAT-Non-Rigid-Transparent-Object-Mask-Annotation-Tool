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

DATASET_PATH = None
image_height = None
image_width = None
predictor = None
js_events = """
<script>
function clickHandler(e) {
    var image_input = document.getElementById("image").querySelector('img');
    if (!image_input) return; // Make sure the image element exists

    var imgWidth = image_input.width;
    var imgHeight = image_input.height;

    var rect = image_input.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
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

def load_scene(scene_id):
    yield f"Loading Scene {scene_id}:"

    scene = SceneFileReader.create(DATASET_PATH / 'config.cfg')

    # Check if masks directory exists
    masks_dir = Path(scene.root_dir) / scene.scenes_dir / scene_id / scene.mask_dir

    if not masks_dir.exists(): 
        yield f"Loading Scene {scene_id}: Generating missing masks"
        vis_masks.create_masks(scene, scene_id)
    else:
        expected_mask_count = len(scene.get_object_poses(scene_id)) * len(scene.get_camera_poses(scene_id))
        if expected_mask_count != len(list(masks_dir.iterdir())):
            gr.Warning(f"Missing masks for scene {scene_id} generating new masks", duration=3)
            yield f"Loading Scene {scene_id}: Generating missing masks"
            vis_masks.create_masks(scene, scene_id)

    yield f"Loaded Scene {scene_id}!"

def click_image(image, evt: gr.SelectData):
    return

def js_trigger(input_data, image):
    data = dict(zip(["x", "y", "button", "imgWidth", "imgHeight"], input_data.split()))
    #TODO factor in image size -> if scaled, need to scale back to original size
    print(data)
    if int(data['x']) < 0 or int(data['x']) > int(data['imgWidth']) or int(data['y']) < 0 or int(data['y']) > int(data['imgHeight']):
        return -1, image
    elif True:
        print("Image Clicked")
        print(image.shape)
        print(image.dtype)
        input_point = np.array([[int(data['x']), int(data['y'])]], dtype=np.int64)
        input_label = np.array([1], dtype=np.int64)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        # for i, (mask, score) in enumerate(zip(masks, scores)):
        #     plt.figure(figsize=(10,10))
        #     plt.imshow(image)
        #     show_mask(mask, plt.gca())
        #     show_points(input_point, input_label, plt.gca())
        #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        #     plt.axis('off')
        #     plt.savefig(f"mask{i}.png")  

        #overlay mask with image using alpha blending
        mask = masks[0]
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = (mask_image * 255).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2RGB)
        mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))
        image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 0)
        return  -1, image
    return -1, image

def main(dataset_path):
    global DATASET_PATH
    global predictor
    DATASET_PATH = dataset_path

    sam_checkpoint = "model_checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # Get list of folders in dataset_path
    scene_folders = sorted([f.stem for f in (dataset_path / 'scenes').iterdir() if f.is_dir()])

    # Load image as np array from path 
    im = cv2.imread(str(dataset_path / 'scenes' / scene_folders[0] / 'rgb' / '000001.png'))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    predictor.set_image(im)

    with gr.Blocks(head=js_events) as demo:
        status_md = gr.Markdown(f"Select a Folder from Dataset {dataset_path}")

        # Create a dropdown to select a scene
        selected_folder = gr.Dropdown(
            choices = scene_folders,
            label = "Select a Scene"
        )
        js_box = gr.Textbox(label="js_parser", elem_id="js_parser", visible=False)
        image_input = gr.Image(label="Upload Image", elem_id="image", value=im)
        selected_folder.change(load_scene, inputs=[selected_folder], outputs=[status_md])
        image_input.select(click_image, [image_input])
        js_box.input(js_trigger, [js_box, image_input], [js_box, image_input])
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