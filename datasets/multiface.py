import os
import torch
import trimesh
import numpy as np
from PIL import Image
'''
            mesh_path = os.path.join(mesh_dir, mesh_name+".obj")
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                mesh_verts = np.asarray(mesh.vertices)
                mesh.compute_vertex_normals()

                T_mesh=np.eye(4)
                with open(os.path.join(mesh_dir, mesh_name+"_transform.txt"), 'r') as f:
                    lines = f.readlines()
                    for lidx, line in enumerate(lines):
                        lsplit = line.split(" ")
                        lsplit[-1] = lsplit[-1][:-1]
                        lsplit = np.asarray(lsplit).astype(np.float32)
                        T_mesh[lidx, :] = lsplit

'''
class MultifaceDataset():
    def __init__(self, data_root, force_overwrite=False):
        self.data_root = data_root 

        self.gather_paths(force_overwrite)

    def gather_paths(self, force_overwrite):
        mesh_paths = []
        pose_paths = []
        texture_paths = []       
        save_dirs = []
        subj_names = sorted(os.listdir(self.data_root))
        for subj_name in subj_names:
            subj_dir = os.path.join(self.data_root, subj_name, "tracked_mesh")
            expr_names = sorted(os.listdir(subj_dir))
            for expr_name in expr_names:
                expr_dir = os.path.join(subj_dir, expr_name)

                file_names = sorted(os.listdir(expr_dir))
                ids = [file_name.split(".")[0] for file_name in file_names if file_name.endswith(".obj")]

                for id in ids:

                    save_dir = expr_dir.replace("tracked_mesh", "registered_mesh")
                    save_dir = os.path.join(save_dir, id)
                    if not force_overwrite:
                        if os.path.exists(save_dir) and len(os.listdir(save_dir)) !=0:                    
                            continue   

                    mesh_paths.append(os.path.join(expr_dir, id+".obj"))
                    pose_paths.append(os.path.join(expr_dir, id+"_transform.txt"))
                    texture_paths.append(os.path.join(subj_dir, "..", "tex_mean.png"))                
                    save_dirs.append(save_dir)                


        self.mesh_paths = mesh_paths
        self.pose_paths = pose_paths
        self.texture_paths = texture_paths
        self.save_dirs = save_dirs   

    def __getitem__(self, index):
        mesh_path = self.mesh_paths[index]
        pose_path = self.pose_paths[index]
        texture_path = self.texture_paths[index]
        save_dir = self.save_dirs[index]        
    
        mesh = trimesh.load(mesh_path, process=False)
        tex_img = Image.open(texture_path).resize((2048, 2048))  # 2k texture map for memory saving
        tex_img = np.asarray(tex_img).astype(np.float32) / 255.0

        verts = np.asarray(mesh.vertices).astype(np.float32)    
        faces = np.asarray(mesh.faces).astype(np.int32)
        normals = np.asarray(mesh.vertex_normals).astype(np.float32)

        T_mesh=np.eye(4)
        with open(pose_path, 'r') as f:
            lines = f.readlines()
            for lidx, line in enumerate(lines):
                lsplit = line.split(" ")
                lsplit[-1] = lsplit[-1][:-1]
                lsplit = np.asarray(lsplit).astype(np.float32)
                T_mesh[lidx, :] = lsplit
        T_mesh = T_mesh.astype(np.float32)
        T_mesh = np.linalg.inv(T_mesh)
        scale = 0.01 * 0.6
        verts = (T_mesh[:3, :3]@verts.transpose(1,0)).transpose(1,0) + T_mesh[:3, -1].reshape(1, 3)        
        normals = (T_mesh[:3, :3]@normals.transpose(1,0)).transpose(1,0)    

        verts = scale * verts
        mesh.vertices = verts

        ###
        # import open3d as o3d
        # test = o3d.geometry.TriangleMesh()
        # test.vertices = o3d.utility.Vector3dVector(verts)
        # test.triangles = o3d.utility.Vector3iVector(faces)
        # test.compute_vertex_normals()

        # ref = o3d.io.read_triangle_mesh("/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/head/merged/nphm/017_001/raw.obj")

        # o3d.visualization.draw_geometries([ref, test])
        ###
        
        uvs = np.asarray(mesh.visual.uv).astype(np.float32)
        uvs[:, 1] = 1 - uvs[:, 1]

        v = torch.from_numpy(verts).cuda()
        n = torch.from_numpy(normals).cuda()
        f = torch.from_numpy(faces).cuda()
        uv = torch.from_numpy(uvs).cuda()
        tex_img = torch.from_numpy(tex_img).cuda()

        return v, n, f, uv, tex_img, save_dir
        
    
    def __len__(self):
        return len(self.mesh_paths)
    



