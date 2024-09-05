import numpy as np
from copy import deepcopy
import time
import pickle
import cv2
import open3d as o3d
import mcubes
import trimesh

class VoxelGrid:
    def __init__(self, width=None, height=None, depth=None, voxel_size=None, origin=None, color=None, img_width=500, img_height=500):
        self.width = width
        self.height = height
        self.depth = depth
        self.voxel_size = voxel_size
        self.origin = origin
        self.color = color
        self.o3d_grid_id = None
        if width is not None and height is not None and depth is not None:
            self.o3d_grid = self.generate_colored_voxelgrid(voxel_size, origin, height, width, depth)
        else:
            self.o3d_grid = None

    def save_voxelgrid(self, path):
        with open(path/"voxel_grid_width.pkl", "wb") as f:
            pickle.dump(self.width, f)
        with open(path/"voxel_grid_height.pkl", "wb") as f:
            pickle.dump(self.height, f)
        with open(path/"voxel_grid_depth.pkl", "wb") as f:
            pickle.dump(self.depth, f)
        with open(path/"voxel_grid_voxel_size.pkl", "wb") as f:
            pickle.dump(self.voxel_size, f)
        with open(path/"voxel_grid_origin.pkl", "wb") as f:
            pickle.dump(self.origin, f)
        with open(path/"voxel_grid_color.pkl", "wb") as f:
            pickle.dump(self.color, f)
        with open(path/"voxel_grid_id.pkl", "wb") as f:
            pickle.dump(self.o3d_grid_id, f)
        
        colors = []
        grid_indexes = []
        for voxel in self.o3d_grid.get_voxels():
            color = voxel.color
            grid_index = voxel.grid_index
            colors.append(color)
            grid_indexes.append(grid_index)
        with open(path/"voxel_grid_colors.pkl", "wb") as f:
            pickle.dump(colors, f)
        with open(path/"voxel_grid_grid_indexes.pkl", "wb") as f:
            pickle.dump(grid_indexes, f)

    def load_voxelgrid(self, path):
        with open(path/"voxel_grid_width.pkl", "rb") as f:
            self.width = pickle.load(f)
        with open(path/"voxel_grid_height.pkl", "rb") as f:
            self.height = pickle.load(f)
        with open(path/"voxel_grid_depth.pkl", "rb") as f:
            self.depth = pickle.load(f)
        with open(path/"voxel_grid_voxel_size.pkl", "rb") as f:
            self.voxel_size = pickle.load(f)
        with open(path/"voxel_grid_origin.pkl", "rb") as f:
            self.origin = pickle.load(f)
        with open(path/"voxel_grid_color.pkl", "rb") as f:
            self.color = pickle.load(f)
        with open(path/"voxel_grid_id.pkl", "rb") as f:
            self.o3d_grid_id = pickle.load(f)

        with open(path/"voxel_grid_colors.pkl", "rb") as f:
            colors = pickle.load(f)
        with open(path/"voxel_grid_grid_indexes.pkl", "rb") as f:
            grid_indexes = pickle.load(f)
        self.o3d_grid = o3d.geometry.VoxelGrid.create_dense(
            origin=self.origin,
            color=self.color,
            voxel_size=self.voxel_size,
            width=self.width,
            height=self.height,
            depth=self.depth
        )

        voxels = self.o3d_grid.get_voxels()
        # remove all voxels
        for voxel in voxels:
            self.o3d_grid.remove_voxel(voxel.grid_index)
        for color, grid_index in zip(colors, grid_indexes):
            voxel = o3d.geometry.Voxel(grid_index, color)
            self.o3d_grid.add_voxel(voxel)

        return self
    
    def show(self, voxel_grid=None):
        if voxel_grid is None:
            voxel_grid = self.o3d_grid
        voxel_grid_visualizer = o3d.visualization.Visualizer()

        voxel_grid_visualizer.create_window()
        voxel_grid_visualizer.add_geometry(voxel_grid)
        voxel_grid_visualizer.run() 

        voxel_grid_visualizer.destroy_window()

    def get_voxel_grid_top_down_view(self, z=1):
        top_down_renderer = o3d.visualization.rendering.OffscreenRenderer(500, 500)

        poi = [self.origin[0] + self.width/2, self.origin[1] + self.height/2, self.origin[2] + self.depth/2]

        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA, does not replace the mesh color
        mtl.shader = "defaultUnlit"

        top_down_renderer.scene.add_geometry("grid", self.o3d_grid, mtl)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(500, 500, 250, 250, 250, 250)
        #extrensics: translation 2 meters above self.poi in z. Looking down
        
        pose = np.array([
            [1, 0, 0, poi[0]],
            [0, 1, 0, poi[1]],
            [0, 0, 1, poi[2]-z],
            [0, 0, 0, 1]
        ])

        extrinsics = np.linalg.inv(pose)

        top_down_renderer.setup_camera(intrinsics, extrinsics)
        img = np.asarray(top_down_renderer.render_to_image())
        return img
    

    def identify_voxels_in_scene(self, scene):
        print("Starting voxel identification...")

        # Initialize camera intrinsics
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            scene.img_width,
            scene.img_height,
            scene.camera_intrinsics[0, 0],
            scene.camera_intrinsics[1, 1],
            scene.camera_intrinsics[0, 2],
            scene.camera_intrinsics[1, 2]
        )
        # Initialize visualizer
        # vis = o3d.visualization.Visualizer()

        num_ids = len(scene.scene_object_ids+scene.get_annotation_object_ids()[0])+1

        # Set of voxels
        voxel_correspondences_global = {tuple(voxel.grid_index): np.zeros(num_ids, dtype=int) for voxel in self.o3d_grid.get_voxels()}

        # Process each annotation image
        for img_idx, image in enumerate(scene.annotation_images.values()):
            if not image.annotation_accepted:
                continue


            # Get camera pose
            pose = image.camera_pose

            # # Project voxel grid into image space
            # vis.create_window(width=scene.img_width, height=scene.img_height, visible=False)
            # vis.add_geometry(self.o3d_grid)
            # view_control = vis.get_view_control()
            # param = o3d.camera.PinholeCameraParameters()
            # param.intrinsic = intrinsics
            # param.extrinsic = np.linalg.inv(pose.tf)
            # view_control.convert_from_pinhole_camera_parameters(param, True)
            # vis.get_render_option().background_color = np.array([0, 0, 0])
            # vis.poll_events()
            # vis.update_renderer()
            # rgb = vis.capture_screen_float_buffer(True)
            # vis.destroy_window()

            rgb = self.project_voxelgrid(scene.img_width, scene.img_height, scene.camera_intrinsics, pose, self.o3d_grid)

            # Convert and scale grid positions
            grid_position = np.array(rgb).astype(np.float32)
            grid_position /= np.array([255, 255, 255])
            grid_position *= np.array([self.width/self.voxel_size, self.height/self.voxel_size, self.depth/self.voxel_size])
            grid_position = np.round(grid_position).astype(np.int32)

            # Remove background pixels and update voxel correspondences
            valid_positions_mask = (grid_position != (0, 0, 0)).all(axis=-1)
            valid_grid_positions = grid_position[valid_positions_mask]
            valid_ids = image.get_complete_segmap()[valid_positions_mask]

            segmap = image.get_complete_segmap()

            # create figure overlaying segmap and grid_position which is an image using matplotlib and save it to disk
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(segmap)
            plt.imshow(grid_position, alpha=0.5)
            plt.savefig(f"segmap_grid_position_{img_idx}.png")
            for pos, voxel_id in zip(valid_grid_positions, valid_ids):
                pos_tuple = tuple(pos)
                if pos_tuple in voxel_correspondences_global:
                    voxel_correspondences_global[pos_tuple][voxel_id] += 1

        # get mean and std of number of votes for each voxel
        votes = np.array([np.sum(votes) for votes in voxel_correspondences_global.values()])
        mean_votes = np.mean(votes)
        std_votes = np.std(votes)

        # Deepcopy the grid for coloring
        colored_voxel_grid = deepcopy(self.o3d_grid)

        # Remove all voxels from the new grid
        for voxel in colored_voxel_grid.get_voxels():
            colored_voxel_grid.remove_voxel(voxel.grid_index)

        # Filter and add colored voxels to the new grid
        for key, value in voxel_correspondences_global.items():
            if (value == 0).all():
                continue
            sorted_ids = np.argsort(value)[::-1]
            if sorted_ids[0] != 0:
                #second most common vote is background: add most dominant object voxel
                if sorted_ids[1] == 0: #and sorted_ids[0] > sorted_ids[1]*1.5:
                    voxel = o3d.geometry.Voxel(key, [sorted_ids[0] / 255] * 3)
                    colored_voxel_grid.add_voxel(voxel)
                # neither of the two most common votes is background: add voxel if it is at least 3 times more common than the second most common vote
                elif sorted_ids[1] != 0 and sorted_ids[0] > sorted_ids[1]*3:
                    voxel = o3d.geometry.Voxel(key, [sorted_ids[0] / 255] * 3)
                    colored_voxel_grid.add_voxel(voxel)
            # elif sorted_ids[0] == 0:
            #     # background is the most common vote: add voxel if the second most common vote is at least 1.5 times less common
            #     if sorted_ids[1]*1.5 >= sorted_ids[0]:
            #         voxel = o3d.geometry.Voxel(key, [sorted_ids[1] / 255] * 3)
            #         colored_voxel_grid.add_voxel(voxel)
            

        # Filter out noise voxels you need at least three neigbours with the same id as the majority vote
        for voxel in colored_voxel_grid.get_voxels():
            grid_index = tuple(voxel.grid_index)
            count = sum(
                np.argmax(voxel_correspondences_global.get((grid_index[0]+i, grid_index[1]+j, grid_index[2]+k), np.zeros(num_ids, dtype=int))) 
                == np.argmax(voxel_correspondences_global[grid_index])
                for i in range(-1, 2)
                for j in range(-1, 2)
                for k in range(-1, 2)
                if not (i == j == k == 0)
            )
            num_votes = np.sum(voxel_correspondences_global[grid_index])
            if num_votes < mean_votes - 3*std_votes:
                if count <= 5:
                    colored_voxel_grid.remove_voxel(grid_index)
            else:
                if count <= 3:
                    colored_voxel_grid.remove_voxel(grid_index)
        self.o3d_grid_id = colored_voxel_grid
    
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

    def project_voxelgrid(self, img_width, img_height, intrinsics, cam_pose=None, voxelgrid=None):
        projection_renderer = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)

        projection_renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
        projection_renderer.scene.view.set_post_processing(False)

        mtl = o3d.visualization.rendering.MaterialRecord()
        # mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA, does not replace the mesh color
        mtl.shader = "defaultUnlit"

        projection_renderer.scene.clear_geometry()
        projection_renderer.scene.add_geometry("grid", voxelgrid, mtl)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        #extrensics: translation 2 meters above self.poi in z. Looking down
        
        pose = cam_pose
        extrinsics = np.linalg.inv(pose.tf)

        projection_renderer.setup_camera(intrinsics, extrinsics)
        img = np.asarray(projection_renderer.render_to_image()).astype(np.uint8)

        # cv2.imshow("img", img*10)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img

    def generate_colored_voxelgrid(self, voxel_size, origin, height, width, depth):
        """
        Optimized version: Creates a voxel grid from a point cloud where each point represents a voxel.
        The point's color encodes its voxel index.

        Args:
            voxel_size: The size of each voxel.
            origin: The origin of the voxel grid.
            height, width, depth: The dimensions of the voxel grid in meters.

        Returns:
            An Open3D voxel grid.
        """

        start_time = time.time()
        self.height = np.ceil(height / voxel_size) * voxel_size
        self.width = np.ceil(width / voxel_size) * voxel_size
        self.depth = np.ceil(depth / voxel_size) * voxel_size

        num_voxels_height = int(self.height / voxel_size)
        num_voxels_width = int(self.width / voxel_size)
        num_voxels_depth = int(self.depth / voxel_size)

        min_bound = origin 

        i, j, k = np.mgrid[:num_voxels_height, :num_voxels_width, :num_voxels_depth]
        print(f"Time taken for initial setup: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        points = min_bound + np.stack([i, j, k], axis=-1) * voxel_size
        points = points.reshape(-1, 3)  # Flatten into a list of points

        colors = np.stack([i, j, k], axis=-1) / np.array([num_voxels_height, num_voxels_width, num_voxels_depth])
        colors = colors.reshape(-1, 3)  # Flatten into a list of colors
        print(f"Time taken for points and colors setup: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)  
        print(f"Time taken for point cloud creation: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)  
        print(f"Time taken for voxel grid creation: {time.time() - start_time:.2f} seconds")

        return voxel_grid
    
    def convert_voxel_grid_to_mesh(self, ids=[]):

        # for ever id in ids create a voxelgrid
        # get voxels from id voxel grid
        # add voxel to corresponding voxel grid
        # create mesh from each voxel_grid

        voxel_grid = np.zeros((int(self.width/self.voxel_size), int(self.height/self.voxel_size), int(self.depth/self.voxel_size)))
        voxel_grids = {k: np.copy(voxel_grid) for k in ids}
        meshes = [] 
        poses = []
        for voxel in self.o3d_grid_id.get_voxels():
            id = np.round(voxel.color[0]*255, 0).astype(int)
            #check if key exists
            if id in voxel_grids:
                voxel_grids[id][voxel.grid_index[0], voxel.grid_index[1], voxel.grid_index[2]] = 15
        
        for id, voxel_grid in voxel_grids.items():
            vertices, triangles = mcubes.marching_cubes(voxel_grid, 0)
            mesh = o3d.geometry.TriangleMesh()
            vertices_transformed =  vertices * self.voxel_size + self.origin
            # mesh.vertices = o3d.utility.Vector3dVector(vertices_transformed)
            # mesh.triangles = o3d.utility.Vector3iVector(triangles)
            # o3d.visualization.draw_geometries([mesh, self.o3d_grid_id])

            # vertices = np.asarray(mesh.vertices)
            # faces = np.asarray(mesh.triangles)
            mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=triangles)

            pose = np.eye(4)
            # pose[:3, 3] = mesh.get_center()
            pose[:3, 3] = np.mean(vertices_transformed, axis=0)
            mesh.apply_transform(np.linalg.inv(pose))

            poses.append(pose)
            meshes.append(mesh)

        return meshes, poses



        # print(voxel_grid.shape)
        # for voxel in self.o3d_grid.get_voxels():
        #     voxel_grid[voxel.grid_index[0], voxel.grid_index[1], voxel.grid_index[2]] = 15
        
        # vertices, triangles = mcubes.marching_cubes(voxel_grid, 0)
        # print(len(vertices), len(triangles))
        # print(vertices[0], triangles[0])
        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(vertices)
        # mesh.triangles = o3d.utility.Vector3iVector(triangles)
        # o3d.visualization.draw_geometries([mesh])
        # print(f"mesh center: mesh.get_center()")
        # triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        # triangle_clusters = np.asarray(triangle_clusters)
        # cluster_n_triangles = np.asarray(cluster_n_triangles)
        # # Get unique cluster labels and corresponding triangle counts
        # unique_labels = np.unique(triangle_clusters)
        # print(unique_labels)
        # # Create separate meshes for each cluster
        # separate_meshes = []
        # for label in unique_labels:
        #     # Filter triangles belonging to the current cluster
        #     cluster_triangles = np.where(triangle_clusters == label)[0]
        #     if len(cluster_triangles) < 100:
        #         continue
        #     mesh_cluster = o3d.geometry.TriangleMesh(
        #         vertices=mesh.vertices,
        #         triangles=o3d.utility.Vector3iVector(
        #             np.asarray(mesh.triangles)[cluster_triangles]
        #         ),
        #     )
        #     separate_meshes.append(mesh_cluster)
        # self.visualize_colored_meshes(separate_meshes)