from utils.v4r import SceneFileReader
from utils.annotationscene import AnnotationScene
import yaml
import pandas as pd
import os
import utils.write_3ddat_to_bop as bop_writer

class AnnotationDataset:
    def __init__(self, dataset_path, config="config.cfg"):
        self.dataset_path = dataset_path
        self.scene_reader = SceneFileReader.create(dataset_path/config)
        self.annotation_scenes = {}
        self.active_scene = None
        self.load_scenes()
        self.object_library = self.yaml_to_dataframe(self.scene_reader.get_object_library_path())


    def load_scenes(self):
        scene_ids = self.scene_reader.get_scene_ids()
        for scene_id in scene_ids:
            camera = self.scene_reader.get_camera_info_scene(scene_id)
            camera_intrinsics = camera.as_numpy3x3()
            self.annotation_scenes[scene_id] = AnnotationScene(scene_id, self.scene_reader, camera_intrinsics, camera.width, camera.height)

    def get_scene_ids(self):
        return self.annotation_scenes.keys()
    
    def yaml_to_dataframe(self, file_path):
        """
        Converts a YAML file containing a list of dictionaries into a Pandas DataFrame.
        
        Args:
            file_path (str): The path to the YAML file.

        Returns:
            pandas.DataFrame: A DataFrame representing the YAML data. If an error occurs (e.g., file not found), returns None.
        """
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)

            # Check if data is a list of dictionaries (the expected format)
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                raise ValueError("Invalid YAML format: Expected a list of dictionaries.")

            # Convert to DataFrame and handle missing values
            df = pd.DataFrame(data)
            df = df.fillna("")  # Replace potential NaN values with empty strings

            # Add full mesh path using file_path as base directory
            # df['mesh'] = df['mesh'].apply(lambda mesh_file: os.path.join(os.path.dirname(file_path), mesh_file))

            return df

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            return None
        except ValueError as e:
            print(e)
            return None
        
    def instanciate_bop_dataset(self, output_path):
        scene_file_reader = self.scene_reader
        # scene_ids = scene_file_reader.get_scene_ids()
        object_lib = scene_file_reader.get_object_library()
    
        OBJ_3D_DAT_TO_BOP_ID = {
        obj_id: int(obj.mesh.file.split('/')[-1][4:-4])
        for obj_id, obj in object_lib.items()
        }

        bop_writer.save_model_information(output_path, OBJ_3D_DAT_TO_BOP_ID, object_lib)