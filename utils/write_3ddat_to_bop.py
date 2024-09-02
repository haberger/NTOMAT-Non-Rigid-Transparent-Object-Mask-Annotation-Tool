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
    def __init__(self, intrinsics, objects, model_colors=None) -> None:
        # --- PyRender scene setup ------------------------------------------------
        self.scene = pyrender.Scene(bg_color=[0, 0, 0])

        if model_colors == None:
            model_colors = [((oi+1), (oi*10) % 255, 200) for oi, _ in enumerate(objects)]
        self.model_colors = model_colors

        # Add model mesh
        for oi, (obj, obj_pose) in enumerate(objects):
            model = deepcopy(obj.mesh.as_trimesh())
            model.apply_transform(np.array(obj_pose).reshape((4, 4)))

            # pyrender render flag SEG does not allow to ignore culling backfaces
            # Instead set color for the mask on the trimesh mesh
            visual = trimesh.visual.create_visual(mesh=model)
            visual.face_colors = self.model_colors[oi]
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

def save_model_information(output_path, OBJ_3D_DAT_TO_BOP_ID, object_lib):
    # === Save models information =============================================
    # create bop tracebot folders
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "models_eval"), exist_ok=True)

    # create model_info for models and models_eval folder bop format
    models_info = dict()
    for oi, (obj_id, obj) in enumerate(object_lib.items()):
        model_path_bop = os.path.join(output_path,
                                 f"models/obj_{OBJ_3D_DAT_TO_BOP_ID[obj_id]:06d}.ply")

        if not os.path.exists(model_path_bop):
            mesh = obj.mesh.as_trimesh()

            # save mesh to bop folder
            # not sure if mesh should be stored in mm or m
            mesh = mesh.apply_scale(1000)  # to mm
            mesh.export(model_path_bop)
            mesh.export(os.path.join(output_path, f"models_eval/obj_{OBJ_3D_DAT_TO_BOP_ID[obj_id]:06d}.ply"))

            # info on dimensions, symmetries, etc
            min_x, min_y, min_z, max_x, max_y, max_z = mesh.bounds.reshape(-1)
            size_x, size_y, size_z = mesh.extents
            samples = trimesh.sample.sample_surface_even(mesh, 2000)[0]
            diameter = np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1).max()
            # models_info[object_id_dict[str(oi + 1)]] = {
            models_info[OBJ_3D_DAT_TO_BOP_ID[obj_id]] = {
                'diameter': diameter,
                'min_x': min_x, 'min_y': min_y, 'min_z': min_z,
                'max_x': max_x, 'max_y': max_y, 'max_z': max_z,
                'size_x': size_x, 'size_y': size_y, 'size_z': size_z,
            }

            # TODO: integrate symmetries information

    if not os.path.exists(f"{output_path}/models/models_info.json"):
        with open(f"{output_path}/models/models_info.json", 'w') as file:
            json.dump(models_info, file, indent=2)
        with open(f"{output_path}/models_eval/models_info.json", 'w') as file:
            json.dump(models_info, file, indent=2)

def write_scene_to_bop(output_path, si, scene_id, scene_file_reader, OBJ_3D_DAT_TO_BOP_ID, mode, flip_upside_down=False, debug=False):

    scene_path_bop = os.path.join(output_path, f"{mode}/{(si+49):06d}")
    scene_path_bop = os.path.join(output_path, f"{mode}/{(si):06d}")
    os.makedirs(scene_path_bop, exist_ok=True)
    os.makedirs(os.path.join(scene_path_bop, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(scene_path_bop, "depth"), exist_ok=True)
    os.makedirs(os.path.join(scene_path_bop, "mask"), exist_ok=True)
    os.makedirs(os.path.join(scene_path_bop, "mask_visib"), exist_ok=True)

    cam_poses_world_cords = scene_file_reader.get_camera_poses(scene_id)
    cam_intrinsics = scene_file_reader.get_camera_info_scene(scene_id)
    object_poses = scene_file_reader.get_object_poses(scene_id)
    sensor_depth_paths = scene_file_reader.get_images_depth_path(scene_id)
    rgb_paths = scene_file_reader.get_images_rgb_path(scene_id)

    cam_intrinsics_final = deepcopy(cam_intrinsics)
    if flip_upside_down:
        cam_intrinsics_final.cx = cam_intrinsics_final.width - cam_intrinsics_final.cx
        cam_intrinsics_final.cy = cam_intrinsics_final.height - cam_intrinsics_final.cy

    with open(os.path.join(output_path, "camera.json"), 'w') as file:
        json.dump({
            'cx': cam_intrinsics_final.cx,
            'cy': cam_intrinsics_final.cy,
            'fx': cam_intrinsics_final.fx,
            'fy': cam_intrinsics_final.fy,
            'height': cam_intrinsics_final.height,
            'width': cam_intrinsics_final.width,
            'depth_scale': 1.0
        }, file, indent=2)


    scene_cameras = dict()
    scene_gts = dict()
    scene_gts_info = dict()

    scene_renderer = SceneRenderer(cam_intrinsics_final, object_poses)
    object_renderers = [
        SceneRenderer(cam_intrinsics_final, [(obj, obj_pose)])
        for (obj, obj_pose) in object_poses
    ]


    for ii, (cam_pose_world, img_path) in enumerate(tqdm(zip(cam_poses_world_cords, rgb_paths), total=len(cam_poses_world_cords))):
        cam_pose_world_final = deepcopy(cam_pose_world)
        if flip_upside_down:
            cam_pose_world_final.tf[:3, :3] = cam_pose_world_final.tf[:3, :3] @ \
                Rotation.from_euler('z', 180, degrees=True).as_matrix()

        img_id = f"{ii:06d}"

        img = cv2.imread(img_path)
        if flip_upside_down:
            img = cv2.rotate(img, cv2.ROTATE_180)

        try:
            depth_path = sensor_depth_paths[ii]
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            if flip_upside_down:
                depth = cv2.rotate(depth, cv2.ROTATE_180)
        except Exception as e:
            print(e)

        cv2.imwrite(os.path.join(scene_path_bop, f"rgb/{img_id}.png"), img)
        cv2.imwrite(os.path.join(scene_path_bop, f"depth/{img_id}.png"), depth)

        # scene camera extrinsics in world coordinates and intrinsics
        cam_R_floats = [float(v) for v in cam_pose_world_final.tf[:3, :3].reshape(-1)]
        cam_t_floats = [float(v) * 1000 for v in cam_pose_world_final.tf[:3, 3].reshape(-1)]

        # prepare and store scene camera to bop
        K = np.array([[cam_intrinsics_final.fx, 0, cam_intrinsics_final.cx],
                        [0, cam_intrinsics_final.fy, cam_intrinsics_final.cy],
                        [0, 0, 1.0]])

        scene_cameras[str(ii)] = {"cam_K": K.reshape(-1).tolist(), "depth_scale": 1.0,
                                            "cam_R_w2c": cam_R_floats, "cam_t_w2c": cam_t_floats}


        instances = dict()

        masks_visible = scene_renderer.render_masks(cam_pose_world_final.tf)
        masks_all = [
            r.render_masks(cam_pose_world_final.tf)[0] for r in object_renderers
        ]

        # TODO: handle training/test mode correctly

        scene_gts[str(ii)] = []
        scene_gts_info[str(ii)] = []
        obj_counter = 0
        for oi, (obj, obj_pose) in enumerate(object_poses):
            obj_id = f"{obj_counter:06d}"

            obj_pose_world_cords = np.array(obj_pose).reshape((4, 4))
            obj_pose_cam = np.linalg.inv(cam_pose_world_final.tf) @ obj_pose_world_cords

            R_floats = [float(v) for v in obj_pose_cam[:3, :3].reshape(-1)]
            t_floats = [float(v) * 1000 for v in obj_pose_cam[:3, 3].reshape(-1)]  # mm


            all_vs, all_us = np.nonzero(masks_all[oi])
            visible_vs, visible_us = np.nonzero(masks_visible[oi])

            if debug:
                viz_img = np.zeros(img.shape, dtype=img.dtype)
                viz_img[masks_visible[oi]] = img[masks_visible[oi]]
                plt.imshow(viz_img[..., ::-1]); plt.show()
                viz_img[masks_all[oi]] = img[masks_all[oi]]
                plt.imshow(viz_img[..., ::-1]); plt.show()

            if len(visible_us) == 0 or len(visible_vs) == 0:
                continue

            obj_info = {
                    "bbox_obj": get_bbox_from_mask(masks_all[oi]),
                    "bbox_visib": get_bbox_from_mask(masks_visible[oi]),
                    "px_count_all": int(len(all_vs)),
                    "px_count_visib": int(len(visible_vs)),
                    "visib_fract": float(1. * len(visible_vs) / len(all_vs))
            }

            try:
                obj_info["px_count_valid"] = int((depth[all_vs, all_us] != 0).sum())
            except Exception as e:
                pass

            cv2.imwrite(os.path.join(scene_path_bop, f"mask_visib/{img_id}_{obj_id}.png"),
                        255 * masks_visible[oi].astype(np.uint8))
            cv2.imwrite(os.path.join(scene_path_bop, f"mask/{img_id}_{obj_id}.png"),
                        255 * masks_all[oi].astype(np.uint8))
            scene_gts[str(ii)].append(
                {"cam_R_m2c": R_floats, "cam_t_m2c": t_floats, "obj_id": OBJ_3D_DAT_TO_BOP_ID[obj.id]}
            )
            scene_gts_info[str(ii)].append(obj_info)
            obj_counter += 1


    # write meta files
    with open(f"{scene_path_bop}/scene_gt.json", 'w') as file:
        json.dump(scene_gts, file, indent=2)
    with open(f"{scene_path_bop}/scene_gt_info.json", 'w') as file:
        json.dump(scene_gts_info, file, indent=2)
    with open(f"{scene_path_bop}/scene_camera.json", 'w') as file:
        json.dump(scene_cameras, file, indent=2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Reproject models to create annotation images.")
    parser.add_argument("-c", "--config", type=str, default="../Dataset/config.cfg",
                        help="Path to reconstructed data")
    parser.add_argument("-o", "--output", type=str, default="../output",
                        help="Path to BOP dataset to create")
    args = parser.parse_args()


    # 3D-DAT scene setup
    scene_file_reader = v4r.SceneFileReader.create(args.config)
    scene_ids = scene_file_reader.get_scene_ids()
    object_lib = scene_file_reader.get_object_library()

    # Map the 3d-dat mesh files to BOP object ids
    # Assumes the filename is in the form "obj_OBJ_ID.ply" with OBJ_ID 00001, etc.
    OBJ_3D_DAT_TO_BOP_ID = {
        obj_id: int(obj.mesh.file.split('/')[-1][4:-4])
        for obj_id, obj in object_lib.items()
    }

    debug = False
    flip_upside_down = False
    mode = "train"                      # train/val
    scenes_to_flip = ['p_018', 'p_019', 'p_020', 'p_021', 'v_002', 'v_003', 'v_004', 'v_013', 'v_014']


    save_model_information(args.output, OBJ_3D_DAT_TO_BOP_ID, object_lib)

    # === Save scene informations =============================================
    for si, scene_id in enumerate(scene_ids):
        write_scene_to_bop(args.output, si, scene_file_reader, OBJ_3D_DAT_TO_BOP_ID, mode)


