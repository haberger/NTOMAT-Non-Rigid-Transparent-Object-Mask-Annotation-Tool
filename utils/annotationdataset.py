from utils.v4r import SceneFileReader
from utils.annotationscene import AnnotationScene

class AnnotationDataset:
    def __init__(self, dataset_path, config="config.cfg"):
        self.dataset_path = dataset_path
        self.scene_reader = SceneFileReader.create(dataset_path/config)
        self.annotation_scenes = {}
        self.active_scene = None
        self.load_scenes()

    def load_scenes(self):
        scene_ids = self.scene_reader.get_scene_ids()
        for scene_id in scene_ids:
            camera = self.scene_reader.get_camera_info_scene(scene_id)
            camera_intrinsics = camera.as_numpy3x3()
            self.annotation_scenes[scene_id] = AnnotationScene(scene_id, self.scene_reader, camera_intrinsics, camera.width, camera.height)

    def get_scene_ids(self):
        return self.annotation_scenes.keys()