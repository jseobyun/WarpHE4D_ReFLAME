import os
import torch
import trimesh
import numpy as np
from PIL import Image
from utils import aa2matrix

class PolygomDataset():
    def __init__(self, data_root, only=-1, force_overwrite=False):
        self.data_root = data_root 
        self.only = only
        self.gather_paths(force_overwrite)

    def gather_paths(self, force_overwrite):
        split_names = [    
            "IOYS_Fullbody_3D스캔_원본이미지_01",
            "IOYS_Fullbody_3D스캔_원본이미지_02",
            "IOYS_Fullbody_3D스캔_원본이미지_03",
            "IOYS_Fullbody_3D스캔_원본이미지_04"
        ]
        if self.only != -1:
            split_names = [split_names[self.only]]

        mesh_paths = []
        texture_paths = []
        flame_param_paths = []
        for split_name in split_names:
            split_dir = os.path.join(self.data_root, split_name)
            subj_names = sorted(os.listdir(split_dir))
            for subj_name in subj_names:
                save_dir = os.path.join(os.path.join(split_dir, subj_name, "meshes"))
                if not force_overwrite:
                    if os.path.exists(save_dir) and len(os.listdir(save_dir)) !=0:                    
                        continue     
                
                mesh_paths.append(os.path.join(split_dir, subj_name, "mesh.obj"))                
                texture_paths.append(os.path.join(split_dir, subj_name, "mesh.jpg"))
                flame_param_paths.append(os.path.join(split_dir, subj_name,  "flame_param.npy"))
                
                                        
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
    



