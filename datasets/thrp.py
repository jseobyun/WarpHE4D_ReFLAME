import os
import torch
import trimesh
import numpy as np
from PIL import Image
from utils import aa2matrix

class THRPDataset():
    def __init__(self, data_root, force_overwrite=False):
        self.data_root = data_root 

        self.gather_paths(force_overwrite)

    def gather_paths(self, force_overwrite):
        mesh_paths = []
        texture_paths = []
        flame_param_paths = []

        subj_names = sorted(os.listdir(self.data_root))
        for subj_name in subj_names:
            save_dir = os.path.join(os.path.join(self.data_root, subj_name, "meshes"))
            if not force_overwrite:
                if os.path.exists(save_dir) and len(os.listdir(save_dir)) !=0:                
                    continue   
            mesh_paths.append(os.path.join(self.data_root, subj_name, subj_name+".obj"))
            texture_path = os.path.join(self.data_root, subj_name, subj_name + ".jpg")
            if not os.path.exists(texture_path):
                texture_path = os.path.join(self.data_root, subj_name, "material_0.jpeg")
            texture_paths.append(texture_path)
            flame_param_paths.append(os.path.join(self.data_root, subj_name,  "flame_param.npy"))

        self.mesh_paths = mesh_paths
        self.texture_paths = texture_paths
        self.flame_param_paths = flame_param_paths
   

    def __getitem__(self, index):
        mesh_path = self.mesh_paths[index]
        texture_path = self.texture_paths[index]
        flame_param_path = self.flame_param_paths[index]
        flame_params = np.load(flame_param_path)[400:]
        R1 = aa2matrix(flame_params[:6]).astype(np.float32)
        R1_inv = np.linalg.inv(R1)
        scale1 = float(flame_params[6])
        trans1 = flame_params[7:].reshape(1, 3).astype(np.float32)
        R1_inv = np.linalg.inv(R1)                        

        mesh = trimesh.load(mesh_path, process=False)
        tex_img = Image.open(texture_path).resize((2048, 2048))  # 2k texture map for memory saving
        tex_img = np.asarray(tex_img).astype(np.float32) / 255.0

        verts = np.asarray(mesh.vertices).astype(np.float32)    
        faces = np.asarray(mesh.faces).astype(np.int32)
        normals = np.asarray(mesh.vertex_normals).astype(np.float32)
        
        verts = (R1_inv @ (verts / scale1 - trans1).transpose(1,0)).transpose(1,0)
        verts = 5*verts
        normals = (R1 @ normals.transpose(1,0)).transpose(1,0)        
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
    



