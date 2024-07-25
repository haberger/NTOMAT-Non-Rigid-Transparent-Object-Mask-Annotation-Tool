import numpy as np
from utils.vis_masks import SceneFileReader
from pathlib import Path
import cv2
import time
import random
import open3d as o3d
from copy import deepcopy
import mcubes

def get_rigit_silhouette(scene):
    # read in all scene masks
    mask_path = Path(scene.root_dir) / scene.scenes_dir / scene_id / scene.mask_dir
    # mask count
    pose_count = len(scene.get_camera_poses(scene_id))
    object_count = len(scene.get_object_poses(scene_id))
    width = scene.get_camera_info_scene(scene_id).width
    height = scene.get_camera_info_scene(scene_id).height
    mask_count = pose_count * object_count
    masks = np.zeros((mask_count, height, width), dtype=np.uint8)
    i = 0
    for mask_file in sorted(mask_path.iterdir()):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        masks[i] = mask
        i += 1

    masks_reshaped = masks.reshape(object_count, pose_count, height, width)
    rigid_silhouettes = np.sum(masks_reshaped, axis=0).astype(np.uint8)
    rigid_silhouettes[rigid_silhouettes > 0] = 255
    return rigid_silhouettes   

def visualize_rays_and_intersection(ois, vis, intersection_point):
    geometries = []

    # Visualize rays
    for oi, vi in zip(ois, vis):
        line_points = [oi, oi + vi * 10]  # Extend the direction vector for visualization
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red color
        geometries.append(line)
    
    # Visualize intersection point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.paint_uniform_color([0, 1, 0])  # Green color
    sphere.translate(intersection_point)
    geometries.append(sphere)
    
    o3d.visualization.draw_geometries(geometries)

def determine_intersection_of_rays(camera_extrinsics, debug_vizualization=False):
    '''adepted from https://math.stackexchange.com/questions/4865611/intersection-closest-point-of-multiple-rays-in-3d-space'''
    
    vis = [pose[:3, 2] for pose in camera_extrinsics]
    ois = [pose[:3, 3] for pose in camera_extrinsics]

    q = np.zeros((3, 3))
    b = np.zeros(3)
    c = 0
    for oi, vi in zip(ois, vis):
        p0 = np.eye(3) - np.outer(vi, vi)
        q += p0
        poi = np.dot(p0, oi) * -2
        b += poi
        c += np.dot(oi, oi)

    try:
        qinv = np.linalg.inv(q)
    except np.linalg.LinAlgError:
        print("Matrix not invertible")
        return None

    x1 = np.dot(qinv, b) * -0.5
    if debug_vizualization:
        visualize_rays_and_intersection(ois, vis, x1)
    return x1

def get_random_color():
    return list(np.random.choice(range(256), size=3) / 255.0)

def visualize_colored_meshes(meshes):
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add each mesh with a unique random color to the visualizer
    for mesh in meshes:
        # Generate and assign random color to the mesh
        color = get_random_color()
        mesh.paint_uniform_color(color)  
        
        # Add the colored mesh to the visualizer
        vis.add_geometry(mesh)

    # Render and display the visualization
    vis.run()
    vis.destroy_window()


def voxel_carving_test(scene, scene_id):
    silhouettes = get_rigit_silhouette(scene)

    # calc obejct poses center
    object_poses = scene.get_object_poses(scene_id)
    translations = []
    for object_pose in object_poses:
        translations.append(object_pose[1][3::4][:3])
    obj_centers = np.mean(translations, axis=0)

    # calc camera poses center
    camera_poses = scene.get_camera_poses(scene_id)
    camera_extrinsics = [camera_pose.tf for camera_pose in camera_poses]
    camera_poi = determine_intersection_of_rays(camera_extrinsics)

    #calc distance between object and camera center
    distance = np.linalg.norm(obj_centers - camera_poi)
    if distance >= 0.5:
        print(f"WARNING: Distance between object and camera center is {distance}m.")
        print(f"Objects should be closer to the camera center.")

    #calc largest distance between cameras and ray intersection
    distances = []
    for camera_pose in camera_extrinsics:
        distances.append(np.linalg.norm(camera_poi - camera_pose[:3, 3]))
    max_distance = np.max(distances)

    width = max_distance * 1.5
    height = max_distance * 1.5
    depth = max_distance * 1.5
    voxel_size = 0.005

    voxel_carving_grid = o3d.geometry.VoxelGrid.create_dense(
        width=width,
        height=height,
        depth=depth,
        voxel_size=voxel_size,
        origin=[camera_poi[0] - width/2, camera_poi[1] - height/2, camera_poi[2] - depth/2],
        color=[0.2, 0.2, 0.2]
    )

    for mask, pose in zip(silhouettes, camera_poses):
        # show mask
        silhouette = o3d.geometry.Image(mask.astype(np.float32))
        extrinsic = np.linalg.inv(pose.tf)
        intrinsic = scene.get_camera_info_scene(scene_id).as_o3d()

        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = intrinsic
        cam.extrinsic = extrinsic

        voxel_carving_grid.carve_silhouette(silhouette, cam, keep_voxels_outside_image=False)
        print(voxel_carving_grid)
        
    o3d.visualization.draw_geometries([voxel_carving_grid])

    # convert voxel grid to numpy array
    voxel_grid = np.zeros((int(width/voxel_size), int(height/voxel_size), int(depth/voxel_size)))
    for voxel in voxel_carving_grid.get_voxels():
        voxel_grid[voxel.grid_index[0], voxel.grid_index[1], voxel.grid_index[2]] = 1
    
    vertices, triangles = mcubes.marching_cubes(voxel_grid, 0)
    mesh = o3d.geometry.TriangleMesh()
    vertices_transformed = vertices * voxel_size + [camera_poi[0] - width/2, camera_poi[1] - height/2, camera_poi[2] - depth/2]
    mesh.vertices = o3d.utility.Vector3dVector(vertices_transformed)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d.visualization.draw_geometries([mesh])

    max_index = np.max(voxel_grid.shape)
    vertex_colors = np.zeros((len(vertices), 3))
    print(vertex_colors.shape)
    for i, vertice in enumerate(vertices):
        vertex_colors[i] = np.array(vertice/max_index)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.visualization.draw_geometries([mesh, voxel_carving_grid])

    # renderer = o3d.visualization.rendering.OffscreenRenderer(500, 500)

    # mtl = o3d.visualization.rendering.MaterialRecord()
    # mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA, does not replace the mesh color
    # mtl.shader = "defaultUnlit"

    # renderer.scene.add_geometry("grid", self.o3d_grid, mtl)

    # intrinsics = o3d.camera.PinholeCameraIntrinsic(500, 500, 250, 250, 250, 250)
    # #extrensics: translation 2 meters above self.poi in z. Looking down
    
    # pose = np.array([
    #     [1, 0, 0, poi[0]],
    #     [0, 1, 0, poi[1]],
    #     [0, 0, 1, poi[2]-z],
    #     [0, 0, 0, 1]
    # ])

    # extrinsics = np.linalg.inv(pose)

    # renderer.setup_camera(intrinsics, extrinsics)
    # img = np.asarray(renderer.render_to_image())

    img_width = scene.get_camera_info_scene(scene_id).width
    img_height = scene.get_camera_info_scene(scene_id).height
    intrinsics = scene.get_camera_info_scene(scene_id).as_o3d()
    extrinsics = np.linalg.inv(camera_poses[0].tf)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    view_control = vis.get_view_control()
    camera_parameters = o3d.camera.PinholeCameraParameters()

    render_option = vis.get_render_option()

    # Disable lighting and shading
    render_option.light_on = False  
    render_option.mesh_show_back_face = True
    # render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.FlatColor

    # Assign intrinsics and extrinsics
    camera_parameters.intrinsic = intrinsics
    camera_parameters.extrinsic = extrinsics
    view_control.convert_from_pinhole_camera_parameters(camera_parameters, True)
    vis.run()
    image = vis.capture_screen_float_buffer(False)  
    image_float = np.asarray(image)   
    vis.destroy_window()
    image = np.asarray(image)
    print(image.dtype)





    for y in range(img_height):
        for x in range(img_width):
            color = image[y, x]
            # if white continue
            if color[0] == 1.0 and color[1] == 1.0 and color[2] == 1.0:
                continue
            for vertex_color in vertex_colors:
                if sum(np.abs(vertex_color-color)) < 0.001:
                    print(vertex_color, color)
                


    # renderer = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)

    # mtl = o3d.visualization.rendering.MaterialRecord()
    # mtl.base_color = [1.0, 1.0, 1.0, 1.0]
    # mtl.shader = "defaultUnlit"

    # renderer.scene.add_geometry("mesh", mesh, mtl)

    # renderer.setup_camera(intrinsics, extrinsics)
    # img = renderer.render_to_image()
    # print(img)
    # print(img.dtype)
    image = np.asarray(image*255).astype(np.uint8)
    cv2.imwrite("voxel_carving.png", image)


def test_voxelgrid_filtering(scene, scene_id):
    silhouettes = get_rigit_silhouette(scene)

    # calc obejct poses center
    object_poses = scene.get_object_poses(scene_id)
    translations = []
    for object_pose in object_poses:
        translations.append(object_pose[1][3::4][:3])
    obj_centers = np.mean(translations, axis=0)

    # calc camera poses center
    camera_poses = scene.get_camera_poses(scene_id)
    camera_extrinsics = [camera_pose.tf for camera_pose in camera_poses]
    camera_poi = determine_intersection_of_rays(camera_extrinsics, debug_vizualization=True)

    #calc distance between object and camera center
    distance = np.linalg.norm(obj_centers - camera_poi)
    if distance >= 0.5:
        print(f"WARNING: Distance between object and camera center is {distance}m.")
        print(f"Objects should be closer to the camera center.")

    #calc largest distance between cameras and ray intersection
    distances = []
    for camera_pose in camera_extrinsics:
        distances.append(np.linalg.norm(camera_poi - camera_pose[:3, 3]))
    max_distance = np.max(distances)

    width = max_distance * 1.5
    height = max_distance * 1.5
    depth = max_distance * 1.5
    voxel_size = 0.005

    voxel_carving_grid = o3d.geometry.VoxelGrid.create_dense(
        width=width,
        height=height,
        depth=depth,
        voxel_size=voxel_size,
        origin=[camera_poi[0] - width/2, camera_poi[1] - height/2, camera_poi[2] - depth/2],
        color=[0.2, 0.2, 0.2]
    )
    masking_grid = deepcopy(voxel_carving_grid)


    #get voxel grid with only relevant points:

    print("prefitlering")

    relevant_points = []
    i = 0
    j=0
    for mask, pose in zip(silhouettes, camera_poses):
        if i%1 == 0:
            mask_grid = deepcopy(masking_grid)
            if j < 3:
                mask = np.ones_like(mask)
                silhouette = o3d.geometry.Image(mask.astype(np.float32))
                extrinsic = np.linalg.inv(pose.tf)
                intrinsic = scene.get_camera_info_scene(scene_id).as_o3d()
                
                cam = o3d.camera.PinholeCameraParameters()
                cam.intrinsic = intrinsic
                cam.extrinsic = extrinsic
                mask_grid.carve_silhouette(silhouette, cam, keep_voxels_outside_image=False)
                relevant_points += [voxel.grid_index for voxel in mask_grid.get_voxels()]
            
                print(mask_grid)
            else:
                break
            j += 1
        i += 1
            
    for voxel in masking_grid.get_voxels():
        masking_grid.remove_voxel(voxel.grid_index)
    for pos in relevant_points:
        new_voxel = o3d.geometry.Voxel(pos, [0, 0, 1])
        masking_grid.add_voxel(new_voxel)
    voxel_carving_grid = masking_grid
    o3d.visualization.draw_geometries([voxel_carving_grid])

    print("carving")

    #do actual voxel carving
    for mask, pose in zip(silhouettes, camera_poses):
        # show mask
        silhouette = o3d.geometry.Image(mask.astype(np.float32))
        extrinsic = np.linalg.inv(pose.tf)
        intrinsic = scene.get_camera_info_scene(scene_id).as_o3d()

        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = intrinsic
        cam.extrinsic = extrinsic
        voxel_carving_grid.carve_silhouette(silhouette, cam, keep_voxels_outside_image=True)
        print(voxel_carving_grid)

    o3d.visualization.draw_geometries([voxel_carving_grid])

if __name__ == "__main__":

    DATASET_PATH = Path('../depth-estimation-of-transparent-objects')
    scene_id = 'j_005'
    scene = SceneFileReader.create(DATASET_PATH / 'config.cfg')
    voxel_carving_test(scene, scene_id)
