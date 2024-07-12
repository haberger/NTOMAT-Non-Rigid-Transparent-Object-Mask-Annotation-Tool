import gradio as gr
import argparse
from pathlib import Path
import time
import utils.vis_masks as vis_masks
from utils.v4r import SceneFileReader
import os
import yaml

DATASET_PATH = None

def load_scene(scene_id):
    yield f"Loading Scene {scene_id}:"

    scene = SceneFileReader.create(DATASET_PATH / 'config.cfg')

    #check if masks directory exists
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



def main(dataset_path):
    global DATASET_PATH
    DATASET_PATH = dataset_path
    #get list of folder in dataset_path
    scene_folders = sorted([f.stem for f in (dataset_path / 'scenes').iterdir() if f.is_dir()])

    with gr.Blocks() as demo:
        status_md = gr.Markdown(f"Select a Folder from Dataset {dataset_path}")

        # create a dropdown to select a scene
        selected_folder = gr.Dropdown(
            choices = scene_folders,
            label = "Select a Scene"
        )
        selected_folder.change(load_scene, inputs=[selected_folder], outputs=[status_md])
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