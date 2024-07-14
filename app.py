import gradio as gr
import argparse
from pathlib import Path
import time
import utils.vis_masks as vis_masks
from utils.v4r import SceneFileReader
import os
import yaml
import cv2

DATASET_PATH = None

shortcut_js = """
<script>
function clickHandler(e) {
    var textbox = document.querySelector('textarea');
    var image_input = document.querySelector('img');
    if (!image_input) return; // Make sure the image element exists
    var rect = image_input.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    var button_label = e.button == 2 ? "right" : "left";
    textbox.value = `Clicked at (${Math.round(x)}, ${Math.round(y)}) with ${button_label} button`;
    // Update the Gradio component with the click data
    gradioApp().getComponentById('image_input').handleSelect({index: [x, y], value: button_label});
    e.preventDefault(); // Prevent the context menu from appearing
}
document.addEventListener('mousedown', clickHandler, false);
document.addEventListener('contextmenu', function(e) {
    e.preventDefault(); // Prevent the context menu from appearing
}, false);
</script>
"""

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

def get_click_data(image, evt: gr.SelectData):
    print("HI")
    return

def main(dataset_path):
    global DATASET_PATH
    DATASET_PATH = dataset_path
    # Get list of folders in dataset_path
    scene_folders = sorted([f.stem for f in (dataset_path / 'scenes').iterdir() if f.is_dir()])

    # Load image as np array from path 
    im = cv2.imread(str(dataset_path / 'scenes' / scene_folders[0] / 'rgb' / '000001.png'))

    with gr.Blocks(head=shortcut_js) as demo:
        status_md = gr.Markdown(f"Select a Folder from Dataset {dataset_path}")

        # Create a dropdown to select a scene
        selected_folder = gr.Dropdown(
            choices = scene_folders,
            label = "Select a Scene"
        )
        output_text = gr.Textbox(label="Click Coordinates (and Segmentation)")
        image_input = gr.Image(label="Upload Image", elem_id="image_input")
        selected_folder.change(load_scene, inputs=[selected_folder], outputs=[status_md])
        image_input.select(get_click_data, [image_input])
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