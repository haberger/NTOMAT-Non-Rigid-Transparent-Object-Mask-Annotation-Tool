import argparse
import cv2
from scipy.spatial.transform import Rotation
import numpy as np
import trimesh
import sys
import os
import json
from tqdm import tqdm
from copy import deepcopy

cwd = os.getcwd()
bop_toolkit_path = os.path.join(cwd,"bop_toolkit")
sys.path.append(bop_toolkit_path)
sys.path.append(os.path.join(bop_toolkit_path, "bop_toolkit_lib"))

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import utils.v4r as v4r

# import utils.pyrender_wrapper
import pyrender

import matplotlib.pyplot as plt

groundtruth_to_pyrender = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])

class SceneRenderer():
    def __init__(self, intrinsics, obj_meshes, obj_ids, obj_poses, model_colors=None) -> None:
        # --- PyRender scene setup ------------------------------------------------
        self.scene = pyrender.Scene(bg_color=[0, 0, 0])

        if model_colors == None:
            model_colors = [((oi), ((oi-1)*10) % 255, 200) for oi in obj_ids]
        self.model_colors = model_colors

        # Add model mesh
        for oi, model, obj_pose in zip(obj_ids, obj_meshes, obj_poses):
            model.apply_transform(np.array(obj_pose).reshape((4, 4)))

            # pyrender render flag SEG does not allow to ignore culling backfaces
            # Instead set color for the mask on the trimesh mesh
            visual = trimesh.visual.create_visual(mesh=model)
            visual.face_colors = self.model_colors[oi-1] # TODO index out of range 
            model.visual = visual
            pyr_mesh = pyrender.Mesh.from_trimesh(model, smooth=False)
            nm = pyrender.Node(mesh=pyr_mesh)
            self.scene.add_node(nm)

        # Add camera
        camera = pyrender.camera.IntrinsicsCamera(intrinsics.fx,
                                                  intrinsics.fy,
                                                  intrinsics.cx,
                                                  intrinsics.cy)
        self.nc = pyrender.Node(camera=camera, matrix=np.eye(4))
        self.scene.add_node(self.nc)
        self.nl = pyrender.Node(matrix=np.eye(4))
        self.scene.add_node(self.nl)

        # Init renderer
        self.renderer = pyrender.OffscreenRenderer(intrinsics.width, intrinsics.height)

    def render_masks(self, cam_pose):
        cam_pose = cam_pose.dot(groundtruth_to_pyrender)
        # Render
        self.scene.set_pose(self.nc, pose=cam_pose)
        self.scene.set_pose(self.nl, pose=cam_pose)

        img, depth = self.renderer.render(
            self.scene,
            flags=pyrender.RenderFlags.SKIP_CULL_FACES |
                pyrender.RenderFlags.FLAT)

        masks = [
            (img[:,:,0] == color[0])
            for color in self.model_colors
        ]

        return masks


def get_bbox_from_mask(mask):
    vs, us = np.nonzero(mask)
    x = us.min()
    y = vs.min()

    width = us.max() - x
    height = vs.max() - y

    bbox = [x, y, width, height]

    return [int(b) for b in bbox]


def transform_cam_poses_to_matrix_form(df_cam_poses_world):
    camera_poses_in_world_coords = {}

    for i, row in df_cam_poses_world.iterrows():
        trans = np.asarray([[row[1], row[2], row[3], 1.]]).T
        rot3x3 = np.asarray(Rotation.from_quat([row[4], row[5], row[6], row[7]]).as_matrix())
        rot_3x4 = np.vstack((rot3x3, np.zeros((1, rot3x3.shape[0]))))  # projection matrix
        camera_poses_in_world_coords[i] = np.hstack((rot_3x4, trans))

    return camera_poses_in_world_coords


def transform_object_from_world_to_camera(df_cam_poses_world, obj_pose_world_coords, rotate_z_axis_180=False):
    obj_poses_in_camera_coords = {}
    camera_poses_in_world_coords = {}

    for i, row in df_cam_poses_world.iterrows():
        trans = np.asarray([[row[1], row[2], row[3], 1.]]).T

        rot3x3 = np.asarray(Rotation.from_quat([row[4], row[5], row[6], row[7]]).as_matrix())

        rot_3x4 = np.vstack((rot3x3, np.zeros((1, rot3x3.shape[0]))))
        projection_matrix = np.hstack((rot_3x4, trans))
        camera_poses_in_world_coords[i] = projection_matrix

        obj_poses_in_camera_coords[i] = np.linalg.inv(projection_matrix) @ obj_pose_world_coords

    return obj_poses_in_camera_coords, camera_poses_in_world_coords


def overlay_input_img_and_rendering(img, render, color=(0, 0, 255), img_intensity=0.6, render_intensity=0.3):
    # Create complete color image
    color_map = np.zeros(img.shape, np.uint8)
    color_map[:] = color

    # Create binary mask from rendering
    mask = (render > 0).astype(bool)

    # Create a colored mask
    colored_mask = mask * color_map

    # Overlay input image with red 2d object projection mask
    overlay_img = cv2.addWeighted(img, img_intensity, colored_mask, render_intensity, 0.0)

    mask = mask.astype(int) * 255

    return overlay_img, mask


def write_on_image(im, text, org=(60,60), font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=2, color=(0,0,255)):
    im = cv2.putText(im, text, org, font, fontScale, (255,255,255), thickness+1, cv2.LINE_AA)
    im = cv2.putText(im, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return im


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img