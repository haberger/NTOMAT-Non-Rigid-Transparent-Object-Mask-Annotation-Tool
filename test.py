import open3d as o3d
import numpy as np
from utils.vis_masks import SceneFileReader
from pathlib import Path

DATASET_PATH = Path('../depth-estimation-of-transparent-objects')

scene_id = 'j_005'

scene = SceneFileReader.create(DATASET_PATH / 'config.cfg')
intrinsic = scene.get_camera_info_scene(scene_id)
objects = scene.get_object_poses(scene_id)
oriented_models = scene.load_object_models(scene_id)
camera_poses = scene.get_camera_poses(scene_id)
camera_info = scene.get_camera_info_scene(scene_id)

camera_pose = camera_poses[0].tf
print(camera_pose)

vis = o3d.visualization.Visualizer()
vis.create_window()

opt = vis.get_render_option()
opt.mesh_show_back_face = True

for model in oriented_models:
    open3d_model = model.as_open3d
    open3d_model.paint_uniform_color(np.random.rand(3,))
    vis.add_geometry(open3d_model)

ctr = vis.get_view_control()
cam_view = ctr.convert_to_pinhole_camera_parameters()
cam_view.extrinsic = np.linalg.inv(camera_pose)
cam_view.intrinsic = camera_info.as_o3d()
ctr.convert_from_pinhole_camera_parameters(cam_view, True)
vis.run()
vis.destroy_window()