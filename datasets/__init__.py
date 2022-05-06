from .blender import BlenderDataset
from .llff import LLFFDataset
from .llff import KubricDataset
from .llff import HyperNeRFDataset
from .phototourism import PhototourismDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'kubric': KubricDataset,
                'hypernerf': HyperNeRFDataset,
                'phototourism': PhototourismDataset}