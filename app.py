import gradio as gr
import argparse
from pathlib import Path
import time
import cv2
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from annotation import AnnotationObject, AnnotationImage, AnnotationScene, AnnotationDataset

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
    if (e.target.tagName !== 'IMG') return;
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

def load_scene(scene_id, prompting_image):
    global dataset

    prompting_image = gr.Image(label="Upload Image", elem_id="prompting_image", elem_classes="images", visible=True) 

    yield f"Loading Scene {scene_id}:", None, gr.Dropdown(visible=False), prompting_image
    scene = dataset.annotation_scenes[scene_id]
    scene.load_images()

    #check if masks are present
    if not scene.has_correct_number_of_masks():
        gr.Warning(f"Missing masks for scene {scene_id} generating new masks", duration=3)
        yield f"Loading Scene {scene_id}: Generating missing masks", None, gr.Dropdown(visible=False), prompting_image
        scene.generate_masks()

    rgb_imgs = scene.annotation_images.keys()

    img_selection = gr.Dropdown(
        value = list(rgb_imgs)[0], 
        choices = rgb_imgs,
        label = "Select an Image",
        visible=True
    )
    dataset.active_scene = scene
    yield f"Loaded Scene {scene_id}!", img_selection, prompting_image


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

def click_image(image, evt: gr.SelectData):
    return

def js_trigger(input_data, image, annotation_objects_selection):
    global dataset

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

    if active_image.active_object.mask is None:
        masks, scores, logits = predictor.predict(
            point_coords=np.array(input_point),
            point_labels=np.array(input_label),
            multimask_output=True,
        )

        active_image.active_object.prompts.append([int(data['x']), int(data['y'])])
        active_image.active_object.prompts_label.append(int(input_label[0]))
        active_image.active_object.mask = masks[np.argmax(scores), :, :]
        active_image.active_object.logit = logits[np.argmax(scores), :, :]

    else:

        active_image.active_object.prompts.append([int(data['x']), int(data['y'])])
        active_image.active_object.prompts_label.append(int(input_label[0]))

        mask, score, logit = predictor.predict(
            point_coords=np.array(active_image.active_object.prompts),
            point_labels=np.array(active_image.active_object.prompts_label),
            mask_input = active_image.active_object.logit[None, :, :],
            multimask_output=False,
        )

        active_image.active_object.mask = mask[0,:,:]
        active_image.active_object.logit = logit[0,:,:]

    image = active_image.generate_visualization()
    return  -1, image

def add_object(annotation_object_name_tb, image):
    global dataset
    global predictor

    active_scene = dataset.active_scene

    for anno_image in active_scene.annotation_images.values():

        annotation_object = AnnotationObject([], [], None, None, annotation_object_name_tb)

        anno_image.active_object = annotation_object
        anno_image.annotation_objects[annotation_object.label] = annotation_object

    for obj in active_scene.active_image.annotation_objects.values():
        if obj.mask is not None:
            image = active_scene.active_image.generate_visualization()

    radio_options = [obj.label for obj in active_scene.active_image.annotation_objects.values()]

    radio_buttons = gr.Radio(label="Select Object", elem_id="annotation_objects", elem_classes="images", choices=radio_options, visible=True, interactive=True, value=radio_options[-1])

    return  "", radio_buttons, image

def change_annotation_object(annotation_objects_selection):
    global dataset
    global predictor

    active_scene = dataset.active_scene

    active_scene.active_image.active_object = active_scene.active_image.annotation_objects[annotation_objects_selection]

    image = active_scene.active_image.generate_visualization()

    return image

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

    with gr.Blocks(head=js_events) as demo:
        status_md = gr.Markdown(f"Select a Folder from Dataset {dataset_path}")
        # prompting_state = gr.State()

        js_parser = gr.Textbox(label="js_parser", elem_id="js_parser", visible=False)
        # #IDEA maybe hide until scene is selected 

        with gr.Row():
            folder_selection = gr.Dropdown(
                choices = dataset.get_scene_ids(),
                label = "Select a Scene",
            )
            img_selection = gr.Dropdown(
                choices = [],
                label = "Select an Image",
                visible=False,
            )

        with gr.Row():
            with gr.Column(scale=5):
                prompting_image = gr.Image(label="Upload Image", elem_id="prompting_image", elem_classes="images", visible=False) 
            with gr.Column(scale=1):
                annotation_object_name_tb = gr.Textbox(label="Object you want to annotate", elem_id="prompting_text", elem_classes="images", visible=True)
                add_annotation_object_btn = gr.Button("Add Object", elem_id="add_annotation_object", elem_classes="images", visible=True)
                annotation_objects_selection = gr.Radio(label="Select Object", elem_id="annotation_objects", elem_classes="images", visible=True)

        folder_selection.change(load_scene, inputs=[folder_selection, prompting_image], outputs=[status_md, img_selection, prompting_image])
        img_selection.change(change_image, inputs=[img_selection], outputs=[prompting_image])
        prompting_image.select(click_image, [prompting_image])
        add_annotation_object_btn.click(add_object, [annotation_object_name_tb, prompting_image], [annotation_object_name_tb, annotation_objects_selection, prompting_image])
        annotation_objects_selection.change(change_annotation_object, [annotation_objects_selection], [prompting_image])
        js_parser.input(js_trigger, [js_parser, prompting_image, annotation_objects_selection], [js_parser, prompting_image])
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