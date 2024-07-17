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


    rgb_imgs_path = scene.scene_reader.get_images_rgb_path(scene_id)

    img_selection = gr.Dropdown(
        value = rgb_imgs_path[0].name, 
        choices = [f.name for f in rgb_imgs_path],
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

def js_trigger(input_data, image):
    global dataset
    data = dict(zip(["x", "y", "button", "imgWidth", "imgHeight"], input_data.split()))
    print(data)
    #TODO factor in image size -> if scaled, need to scale back to original size

    print("Image Clicked")

    input_point = [[int(data['x']), int(data['y'])]]
    input_label = [0] if data['button'] == 'right' else [1]

    active_image = dataset.active_scene.active_image

    if active_image.active_object is None:
        masks, scores, logits = predictor.predict(
            point_coords=np.array(input_point),
            point_labels=np.array(input_label),
            multimask_output=True,
        )
        
        best_mask = masks[np.argmax(scores), :, :]
        best_logit = logits[np.argmax(scores), :, :]
        annotation_object = AnnotationObject(input_point, input_label, best_mask, best_logit, "001")
        active_image.annotation_objects[annotation_object.label] = annotation_object
        active_image.active_object = annotation_object
    else:
        annotation_object = active_image.active_object

        # input_point = np.array([[500, 375], [1125, 625]])
        # input_label = np.array([1, 1])

        annotation_object.prompts.append([int(data['x']), int(data['y'])])
        annotation_object.prompts_label.append(int(input_label[0]))

        print(annotation_object.prompts)
        print(annotation_object.prompts_label)
        print(annotation_object.logit.shape)
        mask, score, logit = predictor.predict(
            point_coords=np.array(annotation_object.prompts),
            point_labels=np.array(annotation_object.prompts_label),
            mask_input = annotation_object.logit[None, :, :],
            multimask_output=False,
        )

        annotation_object.mask = mask[0,:,:]
        annotation_object.logit = logit[0,:,:]
        active_image.active_object = annotation_object

    image = active_image.generate_visualization()
    return  -1, image

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
        prompting_image = gr.Image(label="Upload Image", elem_id="prompting_image", elem_classes="images", visible=False) 
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
        folder_selection.change(load_scene, inputs=[folder_selection, prompting_image], outputs=[status_md, img_selection, prompting_image])
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