
import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import trimesh
import yaml
import open3d as o3d
import copy
from utils.v4r import SceneFileReader

if 'pyrender' in sys.modules:
    raise ImportError(
        'The mesh_to_sdf package must be imported before pyrender is imported.')
if 'OpenGL' in sys.modules:
    raise ImportError(
        'The mesh_to_sdf package must be imported before OpenGL is imported.')

# Disable antialiasing:
import OpenGL.GL

suppress_multisampling = True
old_gl_enable = OpenGL.GL.glEnable

def new_gl_enable(value):
    if suppress_multisampling and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)


OpenGL.GL.glEnable = new_gl_enable

old_glRenderbufferStorageMultisample = OpenGL.GL.glRenderbufferStorageMultisample

def new_glRenderbufferStorageMultisample(target, samples, internalformat, width, height):
    if suppress_multisampling:
        OpenGL.GL.glRenderbufferStorage(target, internalformat, width, height)
    else:
        old_glRenderbufferStorageMultisample(
            target, samples, internalformat, width, height)


OpenGL.GL.glRenderbufferStorageMultisample = new_glRenderbufferStorageMultisample

import pyrender
from PIL import Image

groundtruth_to_pyrender = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])


def project_mesh_to_2d(models, cam_poses, model_colors, intrinsic):
    # --- PyRender scene setup ------------------------------------------------
    scene = pyrender.Scene(bg_color=[0, 0, 0])

    seg_node_map = {}
    # Add model mesh
    for model_idx, model in enumerate(models):
        # pyrender render flag SEG does not allow to ignore culling backfaces
        # Instead set color for the mask on the trimesh mesh
        visual = trimesh.visual.create_visual(mesh=model)
        visual.face_colors = model_colors[model_idx]
        model.visual = visual
        pyr_mesh = pyrender.Mesh.from_trimesh(model, smooth=False)
        nm = pyrender.Node(mesh=pyr_mesh)
        scene.add_node(nm)

    # Add camera
    camera = pyrender.camera.IntrinsicsCamera(intrinsic.fx,
                                              intrinsic.fy,
                                              intrinsic.cx,
                                              intrinsic.cy)
    nc = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(nc)
    nl = pyrender.Node(matrix=np.eye(4))
    scene.add_node(nl)

   # --- Rendering -----------------------------------------------------------
    renders = []
    r = pyrender.OffscreenRenderer(intrinsic.width, intrinsic.height)
    for cam_pose in tqdm(cam_poses, desc="Reprojection rendering"):
        # different coordinate system when using renderer
        cam_pose = cam_pose.dot(groundtruth_to_pyrender)
        # Render
        scene.set_pose(nc, pose=cam_pose)
        scene.set_pose(nl, pose=cam_pose)

        img, depth = r.render(
            scene,
            flags=pyrender.RenderFlags.SKIP_CULL_FACES |
            pyrender.RenderFlags.FLAT)
        renders.append(img)
    return renders


def get_masks_from_render(colors, image):
    masks = []
    img = image
    for idx, color in enumerate(colors):
        np_color = np.array(color)
        if np.shape(image)[2] == 4:
            img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (img == np_color).all(-1)
        masks.append(mask)
    return masks


def get_bbox_from_masks(masks):
    bboxes = []
    for mask in masks:
        if np.any(mask):
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bboxes.append((cmin, rmin, cmax, rmax))
        else:
            bboxes.append(None)
    return bboxes

def put_text(text, img, x, y, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)

    img = cv2.rectangle(img, (x, y - 30), (x + w, y), color, -1)
    img = cv2.putText(img, text, (x, y - 8),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 1)



def create_masks(scene_file_reader, scene_id, output=None):
    
    if output is None:
        output = os.path.join(
            scene_file_reader.root_dir,
            scene_file_reader.scenes_dir, 
            scene_id, 
            scene_file_reader.mask_dir)

    if not os.path.exists(output):
        print(f"Output path {output} does not exist.")
        #create output directory
        os.makedirs(output)

    camera_poses = scene_file_reader.get_camera_poses(scene_id)
    intrinsic = scene_file_reader.get_camera_info_scene(scene_id)
    objects = scene_file_reader.get_object_poses(scene_id)
    oriented_models = scene_file_reader.load_object_models(scene_id)

    model_colors = []
    for i, object in enumerate(objects):
        i += 1
        model_colors.append([i, 0, 0])

    orig_imgs = scene_file_reader.get_images_rgb(scene_id)
    camera_poses = [pose.tf for pose in camera_poses]
    annotation_imgs = project_mesh_to_2d(
        oriented_models, camera_poses, model_colors, intrinsic)

    filepaths = scene_file_reader.get_images_rgb_path(scene_id)
    pbar = tqdm(enumerate(annotation_imgs), desc=f"Saving")
    for pose_idx, anno_img in pbar:
        if np.shape(anno_img)[2] == 3:
            anno_img = cv2.cvtColor(anno_img, cv2.COLOR_RGB2BGRA)
        masks = get_masks_from_render(model_colors, anno_img)

        for i, mask in enumerate(masks):
            filename = f"{objects[i][0].name}_" + \
                f"{i:03d}_" + os.path.basename(filepaths[pose_idx])
            output_path = os.path.join(
                output, filename)
            mask_image = np.array(mask) * 255
            cv2.imwrite(output_path, mask_image)
            pbar.set_description(f"Saving masks into: {output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        "Reproject models to create annotation images.")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="Path to dataset configuration.")
    parser.add_argument("-s", "--scene_id", type=str, required=True,
                        help="Scene identifier to visualize.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for masked images.")
    args = parser.parse_args()

    main(args.dataset, args.scene_id, args.output)

