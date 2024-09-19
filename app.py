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
import pandas as pd
from copy import deepcopy
from collections import deque
import objgraph

dataset = None
predictor = None
# savestate_stack = deque(maxlen=2)

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

def load_predictor(checkpoint_path="model_checkpoints/sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
    """
    Load the predictor model, sets it to the global variable predictor

    Parameters
    ----------
    checkpoint_path : str, optional
        path to the model checkpoints, by default "model_checkpoints/sam_vit_h_4b8939.pth"
    model_type : str, optional
        model type, by default "vit_h"
    device : str, optional
        device to run the model with, by default "cuda"
    """
    global predictor

    print("load_predictor")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)

    predictor = SamPredictor(sam)

def accept_object_library(obj_library_df):
    """
    Transitions the application to the prompting view. This function is called after the object library has been accepted.

    Returns
    -------
    string
        The updated status message to be displayed in the status Markdown component.

    gr.Column
        The object_library_menu_col component with its visibility set to False. This hides the object library menu.

    gr.Column
        The prompting_image_with_col component with its visibility set to True. This shows the image that the user will be prompted to annotate.

    gr.Column
        The annotation_menu_with_col component with its visibility set to True. This shows the menu for annotation options.

    gr.Row
        The scene_navigation_menu_row component with its visibility set to True. This shows the navigation menu for moving through different scenes.
    
    gr.Dropdown
        The annotation_object_dropdown component with the object classes to select from.
    """

    print("accept_object_library")
    dataset_objects = [f"{obj['name']} - {obj['id']}" for _, obj in obj_library_df.iterrows()]

    return (
        "### Select a Scene",
        gr.Column(visible=False),
        gr.Column(scale=8, visible=True),
        gr.Column(scale=2, visible=True),
        gr.Row(visible=True),
        gr.Dropdown(choices=dataset_objects, label="Objects you want to annotate", visible=True)
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
    """

    print("update_object_library")
    global dataset

    if id in obj_library_df['id'].values:
        gr.Warning("ID already exists", duration=3)
        return gr.Dataframe()
    elif id in [None, ""] or object_name in [None, ""] or object_description in [None, ""]:
        gr.Warning("Please fill all textboxes", duration=3)
        return gr.DataFrame()

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

    print("load_scene")
    global dataset

    annotation_objects_selection = gr.Radio(label="Select Object")

    yield (
        f"### Loading Scene {scene_id}:",
        gr.Dropdown(),
        gr.Radio())
    
    scene = dataset.annotation_scenes[scene_id]
    scene.load_scene_data()

    if not scene.has_correct_number_of_masks():
        gr.Warning(f"Missing masks for scene {scene_id} generating new masks", duration=3)
        yield (
            f"### Loading Scene {scene_id}: Generating missing masks",
            gr.Dropdown(),
            gr.Radio())
        scene.generate_masks()


    yield (
        f"### Loading Scene {scene_id}: Loading Images",
        gr.Dropdown(),
        gr.Radio())
    
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
    annotation_objects_selection = gr.Radio(label="Select Object", choices=radio_options, value=default_value)

    yield (
        f"### Select object to annotate",
        img_selection,
        annotation_objects_selection)

def change_image(img_selection, object_selection):
    """
    Change the image to the selected image, load it into the predictor and return the image visualization

    Parameters
    ----------
    img_selection : str
        The selected image to load provided by the img_selection dropdown
    object_selection : str
        The selected object to load to set as active object provided by the annotation_objects_selection radio buttons

    Returns
    -------
    np.ndarray
        The image visualization showing existing prompts and annotations
    """

    print("change_image")
    global dataset
    global predictor 

    if img_selection is None:
        return None

    prompt_image = dataset.active_scene.annotation_images[img_selection]
    dataset.active_scene.active_image = prompt_image

    image = cv2.cvtColor(cv2.imread(prompt_image.rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    if object_selection is not None:
        prompt_image.active_object = prompt_image.annotation_objects[object_selection]

    if dataset.active_scene.manual_annotation_done and not dataset.active_scene.active_image.annotation_accepted:
        prompt_image.generate_auto_prompts(dataset.active_scene, predictor)

    return prompt_image.generate_visualization()

def add_object(object_name):
    """
    Add an object to the active image

    Parameters
    ----------
    object_name : str
        in the fromat "object_name - object_id" provided by the annotation_objects_selection radio buttons

    Returns
    -------
    gr.Radio
        The updated annotation_objects_selection radio buttons
    str
        The updated status message
    """

    print("add_object")
    global dataset

    active_scene = dataset.active_scene

    object_name, object_dataset_id = object_name.split(" - ")
    active_scene.add_object(object_name, object_dataset_id)

    radio_options = [obj.label for obj in active_scene.active_image.annotation_objects.values()]
        
    default_value = radio_options[-1] if radio_options else None

    radio_buttons = gr.Radio(label="Select Object", choices=radio_options, value=default_value)

    status_md = ("### Click on the image to add prompts (Foreground - Left click, "
                "Background - Right click)\n"
                " - If all objects are in frame click **Accept, all objects in "
                "view** if not click **Accept, not all objects in view**\n"
                " - Once all objects have been in frame fully over multiple images click "
                "**Seen All Objects fully** after accepting the annotation -> instantiates the voxel grid\n"
                " - If the voxel grid has sufficient accuracy click **Manual Annotation Done**")


    return status_md, radio_buttons, None

def change_annotation_object(annotation_objects_selection):
    """
    Change the active object to the selected object, and return the image visualization

    Parameters
    ----------
    annotation_objects_selection : str
        The selected object to load to set as active object provided by the annotation_objects_selection radio buttons

    Returns
    -------
    np.ndarray
        The image visualization showing existing prompts and annotations
    """

    print("change_annotation_object")
    global dataset

    if annotation_objects_selection is None:
        return gr.Image()
        # return dataset.active_scene.active_image.generate_visualization()
    prompt_image = dataset.active_scene.active_image

    prompt_image.active_object = prompt_image.annotation_objects[annotation_objects_selection]

    image = prompt_image.generate_visualization()
    return image

def change_opacity(opacity):
    global dataset
    
    dataset.active_scene.visualization_opacity = opacity
    return dataset.active_scene.active_image.generate_visualization() 

def click_image():
    """
    This function just exists so the curser change gets triggered by gradio when hivering over the image.
    Changes the curser to a crosshair for better accuracy when annotating
    """

    print("click_image")
    return

def js_trigger(input_data, annotation_objects_selection, eraser):
    """
    This function is triggered by a javascript event when the user clicks on the image. 
    It adds a prompt to the image at the clicked location, or erases a prompt if the eraser checkbox is checked.
    If no annotation object is selected, it will show a warning message.

    Parameters
    ----------
    input_data : str
        The data string containing the x and y coordinates of the click, the button pressed, and the image width and height
    image : np.ndarray
        The current prompting image visaliszation
    annotation_objects_selection : string
        The selected object to annotate
    eraser : bool
        If the eraser checkbox is checked

    Returns
    -------
    str
        the reset input_data string for the js_parser textbox
    np.ndarray
        The updated image visualization
    """

    print("js_trigger")

    global dataset
    global predictor

    if annotation_objects_selection is None:
        gr.Warning("Please add an object to annotate first", duration=3)
        return -1, gr.Image()

    data = dict(zip(["x", "y", "button", "imgWidth", "imgHeight"], input_data.split()))

    # Scale coordinates back to the original image size if needed
    if data['imgWidth'] != dataset.active_scene.img_width or data['imgHeight'] != dataset.active_scene.img_height:
        data['x'] = int(data['x']) * dataset.active_scene.img_width / int(data['imgWidth'])
        data['y'] = int(data['y']) * dataset.active_scene.img_height / int(data['imgHeight'])

    input_point = [[int(data['x']), int(data['y'])]]
    input_label = [0] if data['button'] == 'right' else [1]

    active_image = dataset.active_scene.active_image

    if eraser:
        active_image.erase_prompt(input_point, predictor)
    else:
        active_image.add_prompt(input_point, input_label, predictor)

    return -1, active_image.generate_visualization()

def accept_annotation(voxel_image, keep_voxels_outside_image, img_selection, voxel_size):
    """
    Accept the annotation for the current image, and carve the voxel grid to the silhouette of the image
    depending on the keep_voxels_outside_image parameter

    Parameters
    ----------
    voxel_image : np.ndarray
        top down view of the voxel grid
    keep_voxels_outside_image : bool
        if True, keep voxels outside the image, if False, only keep voxels inside the image in carving process
    img_selection : str
        current image selected in the img_selection dropdown

    Returns
    -------
    np.ndarray
        top down view of the voxel grid
    str
        changed image selection after moving to the next image
    """

    print("accept_annotation")
    global dataset
    global predictor
    # global savestate_stack

    active_scene = dataset.active_scene
    active_image = active_scene.active_image
    active_image.annotation_accepted = True

    # if len(savestate_stack) == 2:
    #     left_item = savestate_stack.popleft()
    #     objgraph.show_refs(left_item, filename='objgraph_left_item.png')
    #     del left_item
    # savestate_stack.append(deepcopy(dataset))

    status_md = ("### Click on the image to add prompts (Foreground - Left click, "
                "Background - Right click)\n")
    img_selection = active_scene.next_image_name()
    if active_scene.voxel_grid is not None:
        active_scene.carve_silhouette(
            active_image, 
            keep_voxels_outside_image=keep_voxels_outside_image)
        voxel_image = active_scene.voxel_grid.get_voxel_grid_top_down_view()
        yield status_md, voxel_image, img_selection, gr.Button("Seen All Objects fully", visible=False)
    else:
        if keep_voxels_outside_image == False:
            yield "### Instanciating VoxelGrid", gr.Image(), img_selection, gr.Button(visible=False)
            active_scene.instanciate_voxel_grid_at_poi_fast(trigger_image=active_scene.active_image, voxel_size=voxel_size)
            voxel_image = active_scene.voxel_grid.get_voxel_grid_top_down_view()
            yield status_md, voxel_image, img_selection, gr.Button("Seen All Objects fully", visible=False)
        else:
            yield status_md, gr.Image(), img_selection, gr.Button("Seen All Objects fully", visible=True)

def instanciate_voxel_grid(voxel_size):
    """
    Instanciate the voxel grid at the point of interest, hides the button that triggers the instanciation of the voxel grid

    Parameters
    ----------
    voxel_size : float
        size of the voxels in meters

    Yields
    ------
    gr.Button
        button that triggers the instanciation of the voxel grid
    str
        status message
    np.ndarray
        top down view of the voxel grid
    str
        changed image selection for the img_selection dropdown
    """
    
    print("instanciate_voxel_grid")
    global dataset

    active_scene = dataset.active_scene
    status_md = ("### Click on the image to add prompts (Foreground - Left click, "
                "Background - Right click)\n")
    if active_scene.active_image.annotation_accepted is False:
        gr.Warning("Please accept the annotation first", duration=3)
        yield gr.Button("Seen All Objects fully"), status_md, gr.Image(), active_scene.active_image.rgb_path.name
    else:
        button = gr.Button(visible=False)
        yield button, "### Instanciating Voxel Grid", gr.Image(), active_scene.active_image.rgb_path.name

        active_scene.instanciate_voxel_grid_at_poi_with_prefiltering(voxel_size)


        image = active_scene.voxel_grid.get_voxel_grid_top_down_view()
        yield button, status_md, image, active_scene.next_image_name()

def show_voxel_grid():
    """
    Shows the voxel grid in a 3D viewer if it exists
    """

    print("show_voxel_grid")
    global dataset

    active_scene = dataset.active_scene
    if active_scene.voxel_grid is not None:
        active_scene.voxel_grid.show()
    else:
        gr.Warning("No voxel grid to show", duration=3)

def next_image():
    """
    Move to the next image in the scene

    Returns
    -------
    str
        The name of the next image that gets set into the img_selection dropdown
    """

    print("next_image")

    global dataset
    return  dataset.active_scene.next_image_name()

def manual_annotation_done():
    """
    Set the manual annotation done flag to True, and hide the button that triggers the manual annotation done

    Yields
    ------
    str
        status message
    gr.Button
        button that triggers the manual annotation done
    """

    print("manual_annotation_done")

    global dataset

    active_scene = dataset.active_scene

    yield "### Identifying Voxels in Scene", gr.Button(visible=False), gr.Image()

    active_scene.voxel_grid.identify_voxels_in_scene(active_scene)
    active_scene.manual_annotation_done = True

    status_md = ("### Click on the image to add prompts (Foreground - Left click, "
            "Background - Right click), or erase existing prompts\n")
    


    active_scene.active_image.generate_auto_prompts(active_scene, predictor)
    
    yield status_md, gr.Button("Reidentify Voxels in scene", visible=True), active_scene.active_image.generate_visualization()

def write_annotation_to_bop(output_path):

    global dataset
    # check if all images have been annotated
    for img in dataset.active_scene.annotation_images.values():
        if img.annotation_accepted == False:
            gr.Warning(f"Please accept all annotations first [{img.rgb_path.name} not accepted]", duration=3)
            return
        
    print("write_to_bop")


    # get scene index
    if dataset.active_scene is None:
        gr.Warning("Please select a scene first", duration=3)
        return
    dataset.active_scene.write_to_bop(output_path, mode="train")

    return

def restore_savestate():  #TODO handle manual annotation done
    """
    Restores the application to the previous state. This function is called when the user wants to undo their last action. It updates the dataset to the previous state and updates the visibility of various components in the application.

    Returns
    -------
    tuple
        A tuple containing the updated scene ID, image path, annotation objects selection, voxel image, voxel grid button, and visualization image.
    """
    print("restore_savestate")
    # global savestate_stack
    global dataset

    # if len(savestate_stack) > 1:
    #     savestate = savestate_stack.pop()
    #     dataset = deepcopy(savestate)
    # elif len(savestate_stack) == 1:
    #     gr.Info("Last restore point", duration=1)
    #     savestate = savestate_stack[0]
    # else:
    #     gr.Info("No restore points", duration=1)

    voxel_image, instanciate_voxel_grid_btn = get_voxel_image_and_button(dataset)
    annotation_objects_selection = get_annotation_objects_selection(dataset)

    return (
        dataset.active_scene.scene_id, 
        dataset.active_scene.active_image.rgb_path.name, 
        annotation_objects_selection, 
        voxel_image,
        instanciate_voxel_grid_btn,
        dataset.active_scene.active_image.generate_visualization()
    )

def get_voxel_image_and_button(dataset):
    """
    Generates the voxel image and voxel grid button based on the current state of the dataset.

    Parameters
    ----------
    dataset : Dataset
        The current dataset.

    Returns
    -------
    tuple
        A tuple containing the voxel image and voxel grid button.
    """
    if dataset.active_scene.voxel_grid is not None:
        voxel_image = dataset.active_scene.voxel_grid.get_voxel_grid_top_down_view()
        instanciate_voxel_grid_btn = gr.Button(visible=False)
    else:
        voxel_image = gr.Image(None)
        instanciate_voxel_grid_btn = gr.Button("Seen All Objects fully", visible=True)
    return voxel_image, instanciate_voxel_grid_btn

def get_annotation_objects_selection(dataset):
    """
    Generates the annotation objects selection based on the current state of the dataset.

    Parameters
    ----------
    dataset : Dataset
        The current dataset.

    Returns
    -------
    gr.Radio
        The annotation objects selection radio button.
    """
    if dataset.active_scene.active_image.active_object is not None:
        radio_options = [obj.label for obj in dataset.active_scene.active_image.annotation_objects.values()]
        default_value = dataset.active_scene.active_image.active_object.label
    else:
        radio_options = []
        default_value = None
    return gr.Radio(label="Select Object", choices=radio_options, value=default_value)

def generate_and_write_experiment(path):
    #for all images in current scenes that have no annotation accepted
    # clear prompts and generate auto prompts
    # write to bop in folder path that has the number of accpeted annotations in the folder name something like 000005_00x where 5 is the scene id and x is the number of accepted annotations
    # clear all prompts and annotations for images that have no annotation accepted

    global dataset
    global predictor
    active_scene = dataset.active_scene
    
    if active_scene.voxel_grid is None:
        gr.Warning("Please instanciate the voxel grid first", duration=3)
        return

    num_accepted_annotations = 0
    for img in active_scene.annotation_images.values():
        if img.annotation_accepted == False:
            img.reset_prompts()
            rgb = cv2.cvtColor(cv2.imread(img.rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            predictor.set_image(rgb)
            img.generate_auto_prompts(active_scene, predictor)
        else:
            num_accepted_annotations += 1
    
    print(f"Writing to BOP with {num_accepted_annotations} accepted annotations")

    active_scene.write_to_bop(path, mode="train", experiment=num_accepted_annotations)

    print("Writing done")

    for img in active_scene.annotation_images.values():
        if img.annotation_accepted == False:
            img.reset_prompts()

    image = active_scene.active_image.generate_visualization()
    rgb_image = cv2.cvtColor(cv2.imread(active_scene.active_image.rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb_image)
    return image

def main(dataset_path, voxel_size, output_path, checkpoint_path="model_checkpoints/sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
    global dataset
    global predictor

    dataset = AnnotationDataset(dataset_path)
    dataset.instanciate_bop_dataset(output_path)
    load_predictor(checkpoint_path, model_type, device)

    with gr.Blocks(head=js_events, fill_height=True) as demo:
        status_md = gr.Markdown(
            "### Add all objects classes to the object library first, "
            "click \"Accept object library\" to continue")

        js_parser = gr.Textbox(label="js_parser", elem_id="js_parser", visible=False)

        with gr.Row(visible=False) as scene_navigation_menu_row:
            scene_selection = gr.Dropdown(
                choices = dataset.get_scene_ids(),
                label = "Select a Scene",
                scale = 2
            )
            img_selection = gr.Dropdown(
                choices = [],
                label = "Select an Image",
                scale = 2
            )
            next_img_btn = gr.Button("Next Image", scale=2)
            undo_btn = gr.Button("Undo", scale=1)

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
                    choices = [],
                    label="Object you want to annotate")
                
                annotation_objects_selection = gr.Radio(label="Select Object")
                opacity_slider = gr.Slider(label="Visualization_Opacity", minimum=0, maximum=1, step=0.1, value=0.5)

                seen_all_objects_btn = gr.Button("Seen All Objects fully")

                with gr.Row():
                    with gr.Column(min_width=100):
                        accept_annotation_all_in_view_btn = gr.Button("Accept, all objects in view")
                    with gr.Column(min_width=100):
                        accept_annotation_btn = gr.Button("Accept, not all objects in view\n")

                voxel_image = gr.Image(label="Voxel Grid", interactive=False)
                show_grid_btn = gr.Button("Show Voxel Grid")
                manual_annotation_done_btn = gr.Button("Manual Annotation Done")
                write_to_bop_btn = gr.Button("Write to BOP")
                eraser_checkbox = gr.Checkbox(label="erase prompts")
                generate_and_write_experiment_btn = gr.Button("Generate and Write Experiment")
                
        add_object_to_library_btn.click(
            update_object_library, 
            [id_tb, name_tb, description_tb, obj_library_df], 
            [obj_library_df])

        accept_object_library_btn.click(
            accept_object_library,
            inputs=[obj_library_df],
            outputs=[
                status_md, 
                object_library_menu_col, 
                prompting_image_col, 
                annotation_menu_col, 
                scene_navigation_menu_row, 
                annotation_object_dropdown])

        scene_selection.change(
            load_scene, 
            inputs=[scene_selection], 
            outputs=[status_md, img_selection, annotation_objects_selection])
        
        img_selection.change(
            change_image, 
            inputs=[img_selection, annotation_objects_selection], 
            outputs=[prompting_image],
            show_progress='minimal')
        
        annotation_object_dropdown.select(
            add_object, 
            [annotation_object_dropdown], 
            [status_md, annotation_objects_selection, annotation_object_dropdown],
            show_progress='hidden')
        
        annotation_objects_selection.change(
            change_annotation_object, 
            [annotation_objects_selection], 
            [prompting_image],
            show_progress='minimal')
        
        opacity_slider.change(
            change_opacity,
            [opacity_slider],
            [prompting_image],
            show_progress='hidden')

        prompting_image.select(
            click_image)
        
        js_parser.input(
            js_trigger, 
            [js_parser, annotation_objects_selection, eraser_checkbox], 
            [js_parser, prompting_image],
            show_progress='minimal')
        
        accept_annotation_btn.click(
            accept_annotation, 
            [voxel_image, gr.State(True), img_selection, gr.State(voxel_size)], 
            [status_md, voxel_image, img_selection, seen_all_objects_btn],
            trigger_mode='once')
        accept_annotation_all_in_view_btn.click(
            accept_annotation, 
            [voxel_image, gr.State(False), img_selection, gr.State(voxel_size)], 
            [status_md, voxel_image, img_selection, seen_all_objects_btn],
            trigger_mode='once')

        seen_all_objects_btn.click(
            instanciate_voxel_grid,
            [gr.State(voxel_size)],
            outputs=[seen_all_objects_btn, status_md, voxel_image, img_selection])

        show_grid_btn.click(show_voxel_grid)


        next_img_btn.click(next_image, outputs=[img_selection])
        
        manual_annotation_done_btn.click(
            manual_annotation_done, 
            outputs=[status_md, manual_annotation_done_btn, prompting_image])

        write_to_bop_btn.click(
            write_annotation_to_bop, gr.State(output_path))



        undo_btn.click(
            restore_savestate,
            outputs=[scene_selection, img_selection, annotation_objects_selection, voxel_image, seen_all_objects_btn, prompting_image]
        )

        generate_and_write_experiment_btn.click(
            generate_and_write_experiment, [gr.State(output_path)], [prompting_image])

        #TODO
        #make filterin a bit more restrictive -> finetune
        #add button to reidentify voxels in scene -> if manual annotation done -> should be ready to test
        #add opacity slider fox masks in voxel grid -> seems to work
        # TODO dictionary returns -> we only need to return the stuff we want to update





        #to scene writing -> 
        #fix eraser bug
        #fix seen all objects fully bug

        # save existing votes -> only calcuate votes for nexxt image, (votes+confidence+total number of votes) -> calculate majority voting from that information
        # dont to filtering every time, 

        # (differenciate between background and not seen voxels in voting) -> remove background keep unseen, 
        # write complete scene to bop format.

        # check for the mask size of each object in auto prompting -> the ones with big masks are pronably wrong(detect background)
        #TODO Flip
    demo.queue()
    demo.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NTOMAT')
    parser.add_argument(  
        '-d',
        dest='dataset_path',
        default = '../Dataset',
        help='path_to_3D-DAT Dataset')
    
    #add argument for voxel_size
    parser.add_argument(
        '-v',
        dest='voxel_size',
        default=0.004,
        help='voxel_size')

    parser.add_argument(
        '-o',
        dest='output_path',
        default = '/media/Data/Data/DavidDylan/tracebotreal',
        help='path_to_output'
    )

    args = parser.parse_args()

    main(Path(args.dataset_path), voxel_size=float(args.voxel_size), output_path=Path(args.output_path))