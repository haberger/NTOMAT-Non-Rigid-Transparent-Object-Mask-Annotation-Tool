import gradio as gr
import argparse
from pathlib import Path
import time
import cv2
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from utils.annotationdataset import AnnotationDataset
from utils.annotationimage import AnnotationImage, AnnotationObject
from utils.voxelgrid import VoxelGrid
import pandas as pd
import pickle

dataset = None
predictor = None

js_events = """
<script>
function getMousePosition(e, image) {
    var rect = image.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    return { x: Math.round(x), y: Math.round(y) };
}

function handleMouseClick(e) {
    console.log(e.target.offsetParent.offsetParent.id);
    const targetImage = e.target;
    const offsetParentId = "prompting_image"; // ID of the target image's offsetParent div

    if (targetImage.tagName !== 'IMG' || (targetImage.offsetParent.id !== offsetParentId && targetImage.offsetParent.offsetParent.id !== offsetParentId)) return;
    console.log("Image clicked");
    var image = e.target;
    var { x, y } = getMousePosition(e, image);

    var buttonLabel = e.button === 2 ? "right" : "left";
    var jsParser = document.getElementById("js_parser").querySelector('textarea');
    jsParser.value = `${x} ${y} ${buttonLabel} ${image.width} ${image.height}`;
    jsParser.dispatchEvent(new Event('input', { bubbles: true }));
}
function handleContextMenu(e) {
    if (e.target.tagName == 'IMG'){
        e.preventDefault();
    }
}
document.addEventListener('mousedown', handleMouseClick, false);
document.addEventListener('contextmenu', handleContextMenu, false);
</script>
"""

def change_image(img_selection):
    global dataset
    global predictor
    print("Image Changed")
    start_time = time.time()    

    if img_selection is None:
        return None

    scene = dataset.active_scene

    if scene.active_image is not None:
        #TODO handle saving of prompter
        pass

    # check if annotations already exist
    scene.active_image = scene.annotation_images[img_selection]

    if scene.active_image.annotation_objects:
        image = scene.active_image.generate_visualization()
    else:
        image = cv2.cvtColor(cv2.imread(scene.active_image.rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    print(f"Image changed in {time.time() - start_time:.2f} seconds")

    return image

def next_image(img_selection):
    global dataset
    global predictor

    scene = dataset.active_scene
    rgb_imgs = scene.annotation_images.keys()
    indx = list(rgb_imgs).index(img_selection)
    new_selection = list(rgb_imgs)[indx+1]

    return new_selection



def click_image(image, evt: gr.SelectData):
    return

def js_trigger(input_data, image, annotation_objects_selection, eraser):
    global dataset
    global predictor

    print(annotation_objects_selection)
    if annotation_objects_selection is None:
        gr.Warning("Please add an object to annotate first", duration=3)
        return -1, image

    data = dict(zip(["x", "y", "button", "imgWidth", "imgHeight"], input_data.split()))
    print(data)

    #factor in image size -> if scaled, need to scale back to original size
    if data['imgWidth'] != dataset.active_scene.img_width or ['imgHeight'] != dataset.active_scene.img_height:
        x = int(data['x']) * dataset.active_scene.img_width / int(data['imgWidth'])
        y = int(data['y']) * dataset.active_scene.img_height / int(data['imgHeight'])
        data['x'] = x
        data['y'] = y
    
    print("Image Clicked")

    input_point = [[int(data['x']), int(data['y'])]]
    input_label = [0] if data['button'] == 'right' else [1]

    active_image = dataset.active_scene.active_image

    if not eraser:
        active_image.add_prompt(input_point, input_label, predictor)

    else:
        active_image.erase_prompt(input_point, predictor)

    image = active_image.generate_visualization()
    return  -1, image

def add_object(object_name, image): #TODO maybe use ID instead of name
    global dataset
    global predictor

    active_scene = dataset.active_scene
    radio_options = [obj.label for obj in active_scene.active_image.annotation_objects.values()]

    i=0
    object_name = f"{object_name}_{i}"
    while object_name in radio_options:
        i+=1
        object_name = f"{object_name.rpartition('_')[0]}_{i}"

    dataset_object_id = dataset.object_library[dataset.object_library['name'] == object_name.rpartition("_")[0]]['id'].values[0]
    scene_object_id = max(active_scene.scene_object_ids)+1

    for anno_image in active_scene.annotation_images.values():


        annotation_object = AnnotationObject([], [], None, None, object_name, dataset_object_id, scene_object_id)

        anno_image.active_object = annotation_object
        anno_image.annotation_objects[annotation_object.label] = annotation_object

    for obj in active_scene.active_image.annotation_objects.values():
        if obj.mask is not None:
            image = active_scene.active_image.generate_visualization()
    radio_options = [obj.label for obj in active_scene.active_image.annotation_objects.values()]
        
    if len(radio_options) > 0:
        default_value = radio_options[-1]
    else:
        default_value = None

    radio_buttons = gr.Radio(label="Select Object", elem_id="annotation_objects", elem_classes="images", choices=radio_options, visible=True, interactive=True, value=default_value)

    return None, radio_buttons, image

def change_annotation_object(annotation_objects_selection):
    global dataset
    global predictor

    if annotation_objects_selection is None:
        return None
    active_scene = dataset.active_scene

    active_scene.active_image.active_object = active_scene.active_image.annotation_objects[annotation_objects_selection]

    image = active_scene.active_image.generate_visualization()

    return image

def instanciate_voxel_grid():
    global dataset
    global predictor

    active_scene = dataset.active_scene
    button = gr.Button(
        "Seen All Objects", 
        elem_id="seen_all_objects", 
        elem_classes="images", 
        visible=False)
    yield button, "Instanciating Voxel Grid", np.zeros((1,1,3))
    active_scene.instanciate_voxel_grid_at_poi(voxel_size=0.005)
    yield button, "Voxel Grid Instanciated", np.zeros((1,1,3))
    image = active_scene.voxel_grid.get_voxel_grid_top_down_view()
    #save image 
    cv2.imwrite("test.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    yield button, "Voxel Grid Instanciated", image

def accept_annotation(voxel_image, keep_voxels_outside_image, img_selection):
    global dataset
    global predictor

    print("Accepting Annotation")
    print(keep_voxels_outside_image)
    active_scene = dataset.active_scene
    active_image = active_scene.active_image
    active_image.annotation_accepted = True
    if active_scene.voxel_grid is not None:
        active_scene.carve_silhouette(
            active_image, 
            keep_voxels_outside_image=keep_voxels_outside_image)
        voxel_image = active_scene.voxel_grid.get_voxel_grid_top_down_view()

    if active_scene.voxel_grid is not None:
        # go to next image automatically
        rgb_imgs = active_scene.annotation_images.keys()
        indx = list(rgb_imgs).index(img_selection)
        new_selection = list(rgb_imgs)[indx+1]

        return voxel_image, new_selection

    return voxel_image, img_selection

def show_voxel_grid():
    global dataset
    global predictor

    active_scene = dataset.active_scene
    if active_scene.voxel_grid is not None:
        o3d.visualization.draw_geometries([active_scene.voxel_grid.o3d_grid])

def manual_annotation_done():
    global dataset
    global predictor

    active_scene = dataset.active_scene

    #write dataset to pickle into debug_data_promptgeneration
    # active_scene.scene_to_pickle("debug_data_promptgeneration")

    active_scene.voxel_grid.identify_voxels_in_scene(active_scene)

    for image in active_scene.annotation_images.values():
        if image.annotation_accepted:
            continue
        rgb = cv2.imread(image.rgb_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        image.generate_auto_prompts(active_scene, predictor)

    eraser_checkbox = gr.Checkbox(
        label="erase prompts", 
        elem_id="eraser", 
        visible=True,
        scale=3)
    return eraser_checkbox

def change_to_prompting_view():
    """
    Change the view to the prompting view, handles visibility of the different Components after the object library has been accepted

    Returns
    -------
    string
        status_md with the new status message
    gr.Column
        object_library_menu_col with visible=False
    gr.Column
        prompting_image with_col visible=True
    gr.Column
        annotation_menu with_col visible=True
    gr.Row
        scene_navigation_menu_row with visible=True
    """

    return (
        "### Select a Scene",
        gr.Column(visible=False),
        gr.Column(scale=8, visible=True),
        gr.Column(scale=2, visible=True),
        gr.Row(visible=True)
    )

def update_object_library(id, object_name, object_description, obj_library_df):
    """
    This function updates the object library with a new object class

    Parameters
    ----------
    id : str
        The unique ID of the object class provided bt the id_tb
    object_name : str
        The name of the object class provided by the name_tb
    object_description : str
        The description of the object class provided by the description_tb
    obj_library_df : pd.DataFrame
        The current object library dataframe

    Returns
    -------
    pd.DataFrame
        The updated object library dataframe
    gr.Dropdown
        A dropdown with the object classes to select from
    """


    global dataset
    global predictor

    return_objs = [
        obj_library_df, 
        gr.Dropdown(choices=list(obj_library_df['name']), label="Object you want to annotate")]

    if id in obj_library_df['id'].values:
        gr.Warning("ID already exists", duration=3)
        return return_objs
    elif id in [None, ""] or object_name in [None, ""] or object_description in [None, ""]:
        gr.Warning("Please fill all textboxes", duration=3)
        return return_objs

    new_row = pd.DataFrame({
        "id": id, 
        "name": object_name, 
        "description": object_description, 
        "mesh": "", 
        "scale": 0.001, 
        "color": [[255,120,120]]})
    
    obj_library_df = pd.concat([obj_library_df, new_row], ignore_index=True)
    dataset.object_library = obj_library_df

    return obj_library_df[['id', 'name', 'description']]

def load_scene(scene_id):
    """
    Load a scene and its images given a scene_id

    Parameters
    ----------
    scene_id : str
        scene_id of the scene to load provided by the scene_selection dropdown

    Yields
    ------
    str
        status message
    gr.Dropdown
        img_selection dropdown with the images of the scene to select from
    gr.Radio
        annotation_objects_selection radio buttons to select the object to annotate
    """
    global dataset
    global predictor

    annotation_objects_selection = gr.Radio(label="Select Object")

    yield (
        f"Loading Scene {scene_id}:",
        None,
        annotation_objects_selection)
    
    scene = dataset.annotation_scenes[scene_id]
    scene.load_scene_data()

    if not scene.has_correct_number_of_masks():
        gr.Warning(f"Missing masks for scene {scene_id} generating new masks", duration=3)
        yield (
            f"Loading Scene {scene_id}: Generating missing masks",
            None,
            annotation_objects_selection)
        scene.generate_masks()


    yield (
        f"Loading Scene {scene_id}: Loading Images",
        None,
        annotation_objects_selection)
    
    scene.load_images()
    rgb_imgs = scene.annotation_images.keys()
    default_img = list(rgb_imgs)[0] 
    img_selection = gr.Dropdown(
        value = default_img, 
        choices = rgb_imgs,
        label = "Select an Image",
        visible=True
    )
    dataset.active_scene = scene
    scene.active_image = scene.annotation_images[default_img]

    radio_options = [obj.label for obj in scene.active_image.annotation_objects.values()]
    if len(radio_options) > 0:
        default_value = radio_options[-1]
    else:
        default_value = None
    annotation_selection = gr.Radio(label="Select Object", choices=radio_options, value=default_value)

    yield (
        f"Select object to annotate",
        img_selection,
        annotation_selection)


def main(dataset_path):
    global dataset
    global predictor

    dataset = AnnotationDataset(dataset_path)

    sam_checkpoint = "model_checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    with gr.Blocks(head=js_events, fill_height=True) as demo:
        status_md = gr.Markdown(
            "### Add all objects classes to the object library first, "
            "click \"Accept object library\" to continue")
        # prompting_state = gr.State()

        js_parser = gr.Textbox(label="js_parser", elem_id="js_parser", visible=False)

        with gr.Row(visible=False) as scene_navigation_menu_row:
            scene_selection = gr.Dropdown(
                choices = dataset.get_scene_ids(),
                label = "Select a Scene",
            )
            img_selection = gr.Dropdown(
                choices = [],
                label = "Select an Image",
            )
            next_img_btn = gr.Button("Next Image")

        with gr.Row():
            with gr.Column(scale=8) as object_library_menu_col:
                obj_library_df = gr.Dataframe(
                    dataset.object_library[["id", "name", "description"]], 
                    interactive=False)
                with gr.Row():
                    id_tb = gr.Textbox(label="ID")
                    name_tb = gr.Textbox(label="Name")
                    description_tb = gr.Textbox(label="Description")
                    add_object_to_library_btn = gr.Button("Add object")
                    accept_object_library_btn = gr.Button("Accept object library")

            with gr.Column(scale=8, visible=False) as prompting_image_col:
                prompting_image = gr.Image(
                    label="Upload Image", 
                    elem_id="prompting_image", 
                    elem_classes="images",
                    interactive=False) 
                
            with gr.Column(scale=2, visible=False) as annotation_menu_col:
                annotation_object_dropdown = gr.Dropdown(
                    choices = list(dataset.object_library['name']),
                    label="Object you want to annotate")
                
                annotation_objects_selection = gr.Radio(label="Select Object")
                seen_all_objects_btn = gr.Button("Seen All Objects fully")

                with gr.Row():
                    with gr.Column(min_width=100):
                        accept_annotation_all_in_view_btn = gr.Button("Accept, all objects in view")
                    with gr.Column(min_width=100):
                        accept_annotation_btn = gr.Button("Accept, not all objects in view\n")

                voxel_image = gr.Image(label="Voxel Grid", interactive=False)
                show_grid_btn = gr.Button("Show Voxel Grid")
                manual_annotation_done_btn = gr.Button("Manual Annotation Done")
                eraser_checkbox = gr.Checkbox(label="erase prompts")
                
        accept_object_library_btn.click(
            change_to_prompting_view,
            outputs=[
                status_md, 
                object_library_menu_col, 
                prompting_image_col, 
                annotation_menu_col, 
                scene_navigation_menu_row])
        
        add_object_to_library_btn.click(
            update_object_library, 
            [id_tb, name_tb, description_tb, obj_library_df], 
            [obj_library_df])

        scene_selection.change(
            load_scene, 
            inputs=[scene_selection], 
            outputs=[status_md, img_selection, annotation_objects_selection])
        
        img_selection.change(
            change_image, 
            inputs=[img_selection], 
            outputs=[prompting_image])
        prompting_image.select(
            click_image, 
            [prompting_image])
        annotation_object_dropdown.select(
            add_object, 
            [annotation_object_dropdown, prompting_image], 
            [annotation_object_dropdown, annotation_objects_selection, prompting_image])
        annotation_objects_selection.change(
            change_annotation_object, 
            [annotation_objects_selection], 
            [prompting_image])
        js_parser.input(
            js_trigger, 
            [js_parser, prompting_image, annotation_objects_selection, eraser_checkbox], 
            [js_parser, prompting_image])
        seen_all_objects_btn.click(
            instanciate_voxel_grid,
            outputs=[seen_all_objects_btn, status_md, voxel_image])
        accept_annotation_btn.click(
            accept_annotation, 
            [voxel_image, gr.State(True), img_selection], 
            [voxel_image, img_selection])
        accept_annotation_all_in_view_btn.click(
            accept_annotation, 
            [voxel_image, gr.State(False), img_selection], 
            [voxel_image, img_selection])
        manual_annotation_done_btn.click(manual_annotation_done, outputs=[eraser_checkbox])
        show_grid_btn.click(show_voxel_grid)
        next_img_btn.click(next_image, [img_selection], [img_selection])
        # accept_object_library_btn.click(
    demo.queue()
    demo.launch()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NTOMAT')
    parser.add_argument(  
        '-d',
        dest='dataset_path',
        default = '../Dataset',
        help='path_to_3D-DAT Dataset')

    args = parser.parse_args()

    main(Path(args.dataset_path))