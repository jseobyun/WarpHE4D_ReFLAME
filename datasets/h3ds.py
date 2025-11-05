import os
import torch
import trimesh
import numpy as np
from PIL import Image

class H3DSDataset():
    def __init__(self, data_root):
        self.data_root = data_root 

        self.gather_paths()

    def gather_paths(self):
        mesh_paths = []
        texture_paths = []

        version_dirs = [
            os.path.join(self.data_root, "h3ds_v0.2"),
            os.path.join(self.data_root, "h3ds_0.4")]
        for version_dir in version_dirs:
            subj_names = sorted(os.listdir(version_dir))
            for subj_name in subj_names:
                save_dir = os.path.join(os.path.join(version_dir, subj_name, "meshes"))
                # if os.path.exists(save_dir) and len(os.listdir(save_dir)) !=0:                    
                #     continue   
                mesh_paths.append(os.path.join(version_dir, subj_name, "full_head.obj"))
                texture_paths.append(os.path.join(version_dir, subj_name, "full_head.png"))

        self.mesh_paths = mesh_paths
        self.texture_paths = texture_paths
   

    def __getitem__(self, index):
        mesh_path = self.mesh_paths[index]
        texture_path = self.texture_paths[index]
        Rx = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, -1]], dtype=np.float32)
    
        mesh = trimesh.load(mesh_path, process=False)
        tex_img = Image.open(texture_path).resize((2048, 2048))  # 2k texture map for memory saving
        tex_img = np.asarray(tex_img).astype(np.float32) / 255.0

        verts = np.asarray(mesh.vertices).astype(np.float32)    
        faces = np.asarray(mesh.faces).astype(np.int32)
        normals = np.asarray(mesh.vertex_normals).astype(np.float32)

        verts *= 0.01
        verts = (Rx@verts.transpose(1,0)).transpose(1,0)    
        verts *= 0.8
        normals = (Rx@normals.transpose(1,0)).transpose(1,0)    
        mesh.vertices = verts

        
        uvs = np.asarray(mesh.visual.uv).astype(np.float32)
        uvs[:, 1] = 1 - uvs[:, 1]

        v = torch.from_numpy(verts).cuda()
        n = torch.from_numpy(normals).cuda()
        f = torch.from_numpy(faces).cuda()
        uv = torch.from_numpy(uvs).cuda()
        tex_img = torch.from_numpy(tex_img).cuda()

        save_dir = os.path.dirname(mesh_path)
        save_dir = os.path.join(save_dir, "meshes")
        return v, n, f, uv, tex_img, save_dir
        
    
    def __len__(self):
        return len(self.mesh_paths)
    



