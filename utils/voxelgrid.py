import open3d as o3d
import numpy as np
from copy import deepcopy
import time
import pickle
import cv2

class VoxelGrid:
    def __init__(self, width=None, height=None, depth=None, voxel_size=None, origin=None, color=None):
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

            start_time = time.time()
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

            #show rgb image
            rgb_cv = (np.asarray(rgb)*255).astype(np.uint8)
            cv2.imshow("rgb", cv2.cvtColor(rgb_cv, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(f"projection time: {time.time()-start_time}")

            grid_position = np.array(rgb)
            grid_position[:,:,0] = grid_position[:,:,0] * (self.width/self.voxel_size)
            grid_position[:,:,1] = grid_position[:,:,1] * (self.height/self.voxel_size)
            grid_position[:,:,2] = grid_position[:,:,2] * (self.depth/self.voxel_size)
            grid_position = np.round(grid_position).astype(np.int32)

            loop_start_time = time.time()

            valid_positions = (grid_position != (0, 0, 0)).all(axis=-1)
            non_zero_ids = (image.get_complete_segmap() != 0)

            valid_positions_flat = valid_positions.ravel()
            non_zero_ids_flat = non_zero_ids.ravel()
            combined_mask = valid_positions_flat & non_zero_ids_flat

            valid_grid_positions = grid_position.reshape(-1, 3)[combined_mask]
            valid_ids = image.get_complete_segmap().ravel()[combined_mask]

            for pos, id in zip(map(tuple, valid_grid_positions), valid_ids):
                if pos in voxel_correspondences:
                    voxel_correspondences[pos].append(id)

            print(f"loop time: {time.time()-loop_start_time}")
            print(f"correspondence time: {time.time()-start_time}")
        #for each position get majority vote

        print("majority voting")
        for position, votes in voxel_correspondences.items():
            if len(votes) == 0:
                continue
            unique, counts = np.unique(votes, return_counts=True)

            if len(unique) == 1:
                majority_vote = unique[0]
                voxel_correspondences[position] = majority_vote
                continue

            #sort unique and counts decreasing
            sorted_idx = np.argsort(-counts)
            unique = unique[sorted_idx]
            counts = counts[sorted_idx]
            majority_vote = unique[0] if counts[0]/counts[1] > 1.5 else 0
            voxel_correspondences[position] = majority_vote
        
        # def mode_filter(values):
        #     """Calculates the mode (most frequent value) of a neighborhood."""
        #     unique, counts = np.unique(values, return_counts=True)
        #     if len(unique) == 1:
        #         return unique[0]
        #     sorted_idx = np.argsort(-counts)
        #     unique = unique[sorted_idx]
        #     counts = counts[sorted_idx]
        #     if counts[0] / counts[1] > 1.5:
        #         if unique[0] != 0:
        #             return unique[0]
        #         elif counts[0] / sum(counts[1:]) > 7:
        #             return 0
        #     return 256
        

        #z buffer
        



        # print("majority voting with neighborhood analysis")

        # # Convert voxel_correspondences to a 3D grid for neighborhood filtering
        # voxel_grid = np.zeros((int(self.width/self.voxel_size), int(self.height/self.voxel_size), int(self.depth/self.voxel_size)), dtype=np.uint32)
        # print(voxel_grid.shape)
        # for position, vote in voxel_correspondences.items():
        #     if len(vote) == 0:
        #         continue
        #     # set it to majority vote
        #     voxel_grid[position] = max(set(vote), key=vote.count)

        # print("get_neighborhood_votes")
        # # Apply neighborhood analysis (3x3x3 window)
        # neighborhood_votes = generic_filter(voxel_grid, mode_filter, size=3, mode='constant', cval=0)

        # print("combine neighborhood votes with majority voting")

        # # Combine neighborhood votes with majority voting
        # for position, votes in voxel_correspondences.items():
        #     if len(votes) == 0:
        #         continue

        #     unique, counts = np.unique(votes, return_counts=True)

        #     if len(unique) == 1:
        #         voxel_correspondences[position] = unique[0]
        #         continue

        #     sorted_idx = np.argsort(-counts)
        #     unique = unique[sorted_idx]
        #     counts = counts[sorted_idx]

        #     neighborhood_vote = neighborhood_votes[position]
        #     if neighborhood_vote != 256:
        #         voxel_correspondences[position] = neighborhood_vote
        #     elif counts[0] / counts[1] > 1.5:  # Consistent votes
        #         voxel_correspondences[position] = unique[0]
        #     else:
        #         voxel_correspondences[position] = 0
        # print("generating colored voxel grid")

        # copy voxel grid and color voxels according to majority vote
        colored_voxel_grid = deepcopy(self.o3d_grid)
        voxel_indices = [voxel.grid_index for voxel in colored_voxel_grid.get_voxels()]
        for voxel_index in voxel_indices:
            colored_voxel_grid.remove_voxel(voxel_index)
            if tuple(voxel_index) in voxel_correspondences:
                object_id = voxel_correspondences[tuple(voxel_index)]
                if object_id == []:
                    voxelcolor = [0, 0, 0]
                else:
                    voxelcolor = [(1/255)*object_id, (1/255)*object_id, (1/255)*object_id]
            else:
                voxelcolor = [0, 0, 0]    
            voxel = o3d.geometry.Voxel(voxel_index, voxelcolor)
            colored_voxel_grid.add_voxel(voxel)
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

        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=False)
        # vis.destroy_window()
        #visualize voxelgrid

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

        #do some majority filtering to smooth out the segmap

        # cv2.imshow("voxelgrid", img*10)
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

        # Round dimensions to the next multiple of voxel_size
        self.height = np.ceil(height / voxel_size) * voxel_size
        self.width = np.ceil(width / voxel_size) * voxel_size
        self.depth = np.ceil(depth / voxel_size) * voxel_size

        # Calculate number of voxels along each dimension
        num_voxels_height = int(self.height / voxel_size)
        num_voxels_width = int(self.width / voxel_size)
        num_voxels_depth = int(self.depth / voxel_size)

        # Calculate voxel grid bounds
        min_bound = origin 

        # Create grid of indices using NumPy's meshgrid
        i, j, k = np.mgrid[:num_voxels_height, :num_voxels_width, :num_voxels_depth]

        # Calculate point positions using vectorized operations
        points = min_bound + np.stack([i, j, k], axis=-1) * voxel_size
        points = points.reshape(-1, 3)  # Flatten into a list of points

        # Calculate colors using vectorized operations and normalize
        colors = np.stack([i, j, k], axis=-1) / np.array([num_voxels_height, num_voxels_width, num_voxels_depth])
        colors = colors.reshape(-1, 3)  # Flatten into a list of colors

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)  


        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)  
        for voxel in voxel_grid.get_voxels():
            inx = np.round((voxel.color*np.array([num_voxels_height, num_voxels_width, num_voxels_depth])),0).astype(np.int32)
            if not (inx[0] == voxel.grid_index[0] and inx[1] == voxel.grid_index[1] and inx[2] == voxel.grid_index[2]):
                print(inx, voxel.grid_index)
        return voxel_grid
