import xml.etree.ElementTree as ET
import os
import torch
import numpy as np
# from scripts.io_ply import read_ply
import imageio
imageio.plugins.freeimage.download()


def load_scene(filepath):
    """
    Load the meshes, envmap and cameras from a scene XML file.
    We assume the file has the same syntax as Mitsuba 2 scenes.

    Parameters
    ----------

    - filepath : os.path
        Path to the XML file to load
    """
    folder, filename = os.path.split(filepath)
    scene_name, ext = os.path.splitext(filename)
    assert ext == ".xml", f"Unexpected file type: '{ext}'"

    tree = ET.parse(filepath)
    root = tree.getroot()

    assert root.tag == 'scene', f"Unknown root type '{root.tag}', expected 'scene'"

    scene_params = {
        "view_mats" : [],
        "res_x" : 448,
        "res_y" : 448,
        "near_clip": 0.1,
        "far_clip": 100.0,
        "fov": 40,
    }

    for plugin in root:
        if plugin.tag == "emitter" and plugin.attrib["type"] == "envmap":
            for prop in plugin:
                if prop.tag == "string" and prop.attrib["name"] == "filename":
                    envmap_path = os.path.join(folder, prop.attrib["value"])
                    envmap = torch.tensor(imageio.imread(envmap_path, format='HDR-FI'), device='cuda')
                    # Add alpha channel
                    alpha = torch.ones((*envmap.shape[:2],1), device='cuda')
                    scene_params["envmap"] = torch.cat((envmap, alpha), dim=-1)
                elif prop.tag == "float" and prop.attrib["name"] == "scale":
                    scene_params["envmap_scale"] = float(prop.attrib["value"])

    assert "envmap" in scene_params.keys(), "Missing envmap"
    return scene_params
