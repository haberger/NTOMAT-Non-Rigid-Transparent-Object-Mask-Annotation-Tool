import open3d as o3d
import numpy as np
import mcubes
from copy import deepcopy
import cv2

class VoxelGrid:
    def __init__(self, width, height, depth, voxel_size, origin, color):
        self.width = width
        self.height = height
        self.depth = depth
        self.voxel_size = voxel_size
        self.origin = origin
        self.color = color
        self.o3d_grid_id = None
        self.o3d_grid = o3d.geometry.VoxelGrid.create_dense(
            origin=self.origin,
            color=self.color,
            voxel_size=self.voxel_size,
            width=self.width,
            height=self.height,
            depth=self.depth
        )

    def get_voxel_grid_top_down_view(self, z=1):
        # o3d.visualization.draw_geometries([self.o3d_grid])
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=False)
        # vis.destroy_window()

        poi = [self.origin[0] + self.width/2, self.origin[1] + self.height/2, self.origin[2] + self.depth/2]
        renderer = o3d.visualization.rendering.OffscreenRenderer(500, 500)

        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA, does not replace the mesh color
        mtl.shader = "defaultUnlit"

        renderer.scene.add_geometry("grid", self.o3d_grid, mtl)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(500, 500, 250, 250, 250, 250)
        #extrensics: translation 2 meters above self.poi in z. Looking down
        
        pose = np.array([
            [1, 0, 0, poi[0]],
            [0, 1, 0, poi[1]],
            [0, 0, 1, poi[2]-z],
            [0, 0, 0, 1]
        ])

        extrinsics = np.linalg.inv(pose)

        renderer.setup_camera(intrinsics, extrinsics)
        img = np.asarray(renderer.render_to_image())
        return img
    
    def identify_voxels_in_scene(self, scene):
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            scene.img_width, 
            scene.img_height, 
            scene.camera_intrinsics[0, 0], 
            scene.camera_intrinsics[1, 1], 
            scene.camera_intrinsics[0, 2], 
            scene.camera_intrinsics[1, 2])
        vis = o3d.visualization.Visualizer()

        voxel_correspondences = {tuple(voxel.grid_index): [] for voxel in self.o3d_grid.get_voxels()}
        #iterate over all images that have accepted annotations
        for image in scene.annotation_images.values():
            if not image.annotation_accepted:
                continue

            print("calculating correspondences for image")

            # get camera pose
            pose = image.camera_pose

            #project voxel grid into image space
            vis.create_window(width=scene.img_width, height=scene.img_height, visible=False)
            vis.add_geometry(self.o3d_grid)
            view_control = vis.get_view_control()
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic = intrinsics
            param.extrinsic = np.linalg.inv(pose.tf)
            view_control.convert_from_pinhole_camera_parameters(param, True)
            #define background color
            vis.get_render_option().background_color = np.array([0, 0, 0])
            vis.poll_events()
            vis.update_renderer()
            # vis.run()
            rgb = vis.capture_screen_float_buffer(True)
            vis.destroy_window()

            # #show rgb image
            # rgb_cv = (np.asarray(rgb)*255).astype(np.uint8)
            # cv2.imshow("rgb", cv2.cvtColor(rgb_cv, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


            grid_position = np.array(rgb)
            grid_position[:,:,0] = grid_position[:,:,0] * (self.width/self.voxel_size)
            grid_position[:,:,1] = grid_position[:,:,1] * (self.height/self.voxel_size)
            grid_position[:,:,2] = grid_position[:,:,2] * (self.depth/self.voxel_size)
            grid_position = np.round(grid_position).astype(np.int32)

            for i in range(scene.img_height):
                for j in range(scene.img_width):
                    
                    position = tuple(grid_position[i, j])
                    if position == (0, 0, 0):
                        continue
                    if position not in voxel_correspondences:
                        continue
                    id = image.segmap[i, j]
                    if id != 0:
                        voxel_correspondences[position].append(image.segmap[i, j])

        #for each position get majority vote

        print("majority voting")
        for position, votes in voxel_correspondences.items():
            if len(votes) == 0:
                continue
            majority_vote = max(set(votes), key=votes.count)
            voxel_correspondences[position] = majority_vote
        print("generating colored voxel grid")

        # copy voxel grid and color voxels according to majority vote
        colored_voxel_grid = deepcopy(self.o3d_grid)
        voxel_indices = [voxel.grid_index for voxel in colored_voxel_grid.get_voxels()]
        for voxel_index in voxel_indices:
            colored_voxel_grid.remove_voxel(voxel_index)
            if tuple(voxel_index) in voxel_correspondences:
                color = voxel_correspondences[tuple(voxel_index)]
                if color == []:
                    voxelcolor = [0, 0, 0]
                else:
                    voxelcolor = [(1/255)*color, (1/255)*color, (1/255)*color]
            else:
                voxelcolor = [0, 0, 0]    
            voxel = o3d.geometry.Voxel(voxel_index, voxelcolor)
            colored_voxel_grid.add_voxel(voxel)
        self.o3d_grid_id = colored_voxel_grid
        print("displaying colored voxel grid")
        o3d.visualization.draw_geometries([colored_voxel_grid])


    
    def visualize_colored_meshes(self, meshes):
        def get_random_color():
            return list(np.random.choice(range(256), size=3) / 255.0)
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        for mesh in meshes:
            color = get_random_color()
            mesh.paint_uniform_color(color)  
            
            vis.add_geometry(mesh)
        vis.add_geometry(self.o3d_grid)
        vis.run()
        vis.destroy_window()

    def convert_voxel_grid_to_mesh(self):

        voxel_grid = np.zeros((int(self.width/self.voxel_size), int(self.height/self.voxel_size), int(self.depth/self.voxel_size)))
        print(voxel_grid.shape)
        for voxel in self.o3d_grid.get_voxels():
            voxel_grid[voxel.grid_index[0], voxel.grid_index[1], voxel.grid_index[2]] = 15
        
        vertices, triangles = mcubes.marching_cubes(voxel_grid, 0)
        print(len(vertices), len(triangles))
        print(vertices[0], triangles[0])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        o3d.visualization.draw_geometries([mesh])
        print(f"mesh center: mesh.get_center()")

        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        # Get unique cluster labels and corresponding triangle counts
        unique_labels = np.unique(triangle_clusters)
        print(unique_labels)

        # Create separate meshes for each cluster
        separate_meshes = []
        for label in unique_labels:
            # Filter triangles belonging to the current cluster
            cluster_triangles = np.where(triangle_clusters == label)[0]
            if len(cluster_triangles) < 100:
                continue
            mesh_cluster = o3d.geometry.TriangleMesh(
                vertices=mesh.vertices,
                triangles=o3d.utility.Vector3iVector(
                    np.asarray(mesh.triangles)[cluster_triangles]
                ),
            )
            separate_meshes.append(mesh_cluster)
        self.visualize_colored_meshes(separate_meshes)

    def project_voxelgrid(self, img_width, img_height, intrinsics, cam_pose=None, voxelgrid=None):

        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=False)
        # vis.destroy_window()
        #visualize voxelgrid
        o3d.visualization.draw_geometries([voxelgrid])

        renderer = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
        renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
        renderer.scene.view.set_post_processing(False)

        mtl = o3d.visualization.rendering.MaterialRecord()
        # mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA, does not replace the mesh color
        mtl.shader = "defaultUnlit"

        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("grid", voxelgrid, mtl)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        #extrensics: translation 2 meters above self.poi in z. Looking down
        
        pose = cam_pose
        print(pose)
        extrinsics = np.linalg.inv(pose.tf)

        renderer.setup_camera(intrinsics, extrinsics)
        img = np.asarray(renderer.render_to_image()).astype(np.uint8)
        return img
