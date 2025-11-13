import os
import cv2
import shutil
import torch
import trimesh
import argparse
import numpy as np
import open3d as o3d
import nvdiffrast.torch as dr
import torch.nn.functional as F
from PIL import Image
from tqdm import trange, tqdm
from templates.FLAME import FLAME
from largesteps.load_xml import load_scene
from largesteps.render import NVDRenderer
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix, compute_vertex_normals, compute_face_normals
from largesteps.optimize import AdamUniform

from utils import (to_np, gen_orbit_views, save_flame_params, total_variation_loss, vertices2landmarks, load_flame_uv, load_p3d_mesh,
                   to_cuda, build_o3d_mesh, FaceKeypointDetector)
from datasets import H3DSDataset, Renderme360Dataset, THRPDataset, PolygomDataset, MergedDataset, MultifaceDataset


def parse_config():
    parser = argparse.ArgumentParser("ReFLAME configuration")    
    ###
    parser.add_argument("--dataset_name", type=str, default="th")
    parser.add_argument("--only", type=int, default=0)
    ###    
    parser.add_argument("--texture_steps", type=int, default=500, help="1K texture map optimization iterations")
    parser.add_argument("--azim_range", type=float, default=170, help="azimuth angle range [-a, a] for renderer. 150 degree recomennded")
    parser.add_argument("--elev_range", type=float, default=30, help="azimuth angle range [-a, a] for renderer. 150 degree recomennded")  
    ### debugging
    parser.add_argument("--vis", action="store_true", default=False, help="enable visualization for debugging")
    parser.add_argument("--save", action="store_true", default=True, help="enable save")
    args = parser.parse_args()

  
    return args



def bake_position_to_uv(v, f, face_uvs, tex_h=768, tex_w=768, glctx=None):
    """
    v:        [V,3]  float32, cuda (geometry vertices)
    f:        [F,3]  int64/int32, cuda (geometry faces)
    face_uvs: [F,3,2] float32, cuda (UV per face vertex, in [0,1])

    return:
        pos_tex: [tex_h, tex_w, 3]
        mask:    [tex_h, tex_w] bool
    """
    device = v.device
    if glctx is None:
        glctx = dr.RasterizeCudaContext(device=device)

    F = f.shape[0]

    # 1) face_uvs -> [F*3,2]
    uv_flat = face_uvs.reshape(-1, 2)   # [F*3,2]

    # 2) UV vertex 인덱스: 0..F*3-1
    uv_tri = torch.arange(F * 3, device=device, dtype=torch.int32).reshape(F, 3)  # [F,3]

    # 3) UV(0~1) -> NDC(-1~1), z=0, w=1
    x = uv_flat[:, 0] * 2.0 - 1.0
    y = uv_flat[:, 1] * 2.0 - 1.0
    z = torch.zeros_like(x)
    w = torch.ones_like(x)

    pos = torch.stack([x, y, z, w], dim=-1)  # [F*3,4]
    pos = pos[None, ...].contiguous()        # [1, F*3,4]  ✅ instance mode OK

    # 4) 각 UV-vertex에 대응되는 geometry attribute 준비
    geom_attr = v[f.reshape(-1)]             # [F*3,3]
    geom_attr = geom_attr[None, ...].contiguous()  # [1,F*3,3]

    # 5) UV plane에서 rasterize
    rast, _ = dr.rasterize(glctx, pos, uv_tri, (tex_h, tex_w))  # rast: [1,H,W,4]

    # 6) geometry를 UV domain으로 bake
    pos_tex, _ = dr.interpolate(geom_attr, rast, uv_tri)  # [1,H,W,3]

    # 유효 픽셀 (tri id >= 0)
    tri_id = rast[0, ..., 3]          # [H,W]
    mask = tri_id > 0

    return pos_tex[0], mask

if __name__ == "__main__":

    args = parse_config()    


    '''
    Load renderer settings : envmap is only for visualization.
    '''

    filepath = "./templates/ENV/environment.xml"
    scene_params = load_scene(filepath)
    scene_params["res_x"] = 768
    scene_params["res_y"] = 768
    T_kgs, T_gks, view_mask = gen_orbit_views(elev_range=args.elev_range, azim_range=args.azim_range, num_gens=8)
    scene_params["view_mats"] = T_kgs

    renderer = NVDRenderer(scene_params, shading=False, boost=3) # check shading False during optimization.

    ### heuristic overwrite 
    dataset_name = args.dataset_name
    if dataset_name in ["nphm", "faceverse", "facescape", "wysiwig"]:
        data_root = "/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/head/merged"
        dataset = MergedDataset(os.path.join(data_root, dataset_name), force_overwrite=True)
        print(f"{dataset_name} are prepared : {len(dataset)}")
    elif "ioys" in dataset_name:
        data_root = "/home/jseob/Desktop/yjs/nas/data/01_IOYS/body_mesh_from_agisoft_v1"
        dataset = PolygomDataset(data_root, args.only, force_overwrite=True)
        print(f"{dataset_name} {args.only} are prepared : {len(dataset)}")
    elif dataset_name in ["th", "rp"]:        
        if dataset_name == "th":
            data_root = "/home/jseob/Desktop/yjs/nas/data/11_BODY/TH2.1/MESH"
        else:
            data_root = "/home/jseob/Desktop/yjs/nas/data/11_BODY/RP/MESH"
        dataset = THRPDataset(data_root, force_overwrite=True)
        print(f"{dataset_name} are prepared : {len(dataset)}")

    elif dataset_name == "h3ds":
        data_root = "/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/head/h3ds"
        dataset = H3DSDataset(data_root, force_overwrite=True)
        print(f"{dataset_name} are prepared : {len(dataset)}")

    elif dataset_name == "renderme":
        data_root = "/media/jseob/X9/renderme360/processed"
        dataset = Renderme360Dataset(data_root, force_overwrite=True)
        print(f"{dataset_name} are prepared : {len(dataset)}")
    elif dataset_name == "multiface":
        data_root = "/media/jseob/7c338ab7-a4a5-460a-a3bb-6c26309b51ba/datasets/head/multiface"
        dataset = MultifaceDataset(data_root, force_overwrite=True)
        print(f"{dataset_name} are prepared : {len(dataset)}")    

    for subj_idx, (v_ref, n_ref, f_ref, uv_ref, tex_img, save_dir) in enumerate(dataset):
        print(f"{save_dir} {subj_idx}/{len(dataset)}")                        
        if True: #try: ### first try
            if args.save:            
                os.makedirs(save_dir, exist_ok=True)        
            '''
            Load reference (target) mesh and FLAME meta info.
            '''
            # v_ref, n_ref, f_ref, uv_ref, tex_img = load_mesh_under_dir(args.data_dir)   

            flame_model = FLAME("./templates/FLAME2023").cuda()
            full_lmk_faces_idx = flame_model.full_lmk_faces_idx
            full_lmk_bary_coords = flame_model.full_lmk_bary_coords   
            vt_np, ft_np, f_np = load_flame_uv("./templates/FLAME2023")

                
            f = to_cuda(f_np)
            vt = to_cuda(vt_np)
            ft = to_cuda(ft_np)

            '''
            Render reference images, masks, and depths
            '''
            ref_imgs, ref_masks, ref_depths = renderer.render(v_ref, n_ref, f_ref, tex_img, uv_ref)
            num_imgs = ref_imgs.size(0)

            '''        
            Setting optimization
            '''               
            nonrigid_flame = load_p3d_mesh(
                os.path.join(save_dir, "flame_nonrigid_mesh_wo_eye.ply"), 
                subdiv=0,
                device="cuda:0") # for double checking existence of flame_rigid_mesh.ply

            v = nonrigid_flame._verts_list[0].detach() # [V, 3]
            f = nonrigid_flame._faces_list[0].detach() # [F, 3]
            face_normals = compute_face_normals(v, f)
            n = compute_vertex_normals(v, f, face_normals)

            vt = np.load('./templates/FLAME2023/flame_subdiv2_verts_uvs_v2.npy')        
            vt = 1 - vt
            vt = torch.from_numpy(vt).cuda() # V' 2
            ft = np.load('./templates/FLAME2023/flame_subdiv2_tex_uvs_v2.npy').astype(np.int32)
            ft = torch.from_numpy(ft).cuda() # F, 3

            face_uvs = vt[ft.reshape(-1)].reshape(-1, 3, 2) # F*3, 2            
            # '''
            # Positionmap optimization
            # '''
            # position_map, position_mask = bake_position_to_uv(v, f, face_uvs)
            # if args.save:
            #     pos_path = os.path.join(save_dir, "flame_position_map.npy")                
            #     position_map_np = to_np(position_map)                
            #     np.save(pos_path, position_map_np)
                       
            '''
            Texture map optimization
            '''
            if os.path.exists(os.path.join(save_dir, "flame_texture.jpg")):
                tex_img = Image.open(os.path.join(save_dir, "flame_texture.jpg")).convert("RGB")
                tex_img = tex_img.resize((768, 768))
                tex_img = np.asarray(tex_img).astype(np.float32)/255.0                
                tex_img = torch.tensor(tex_img, device="cuda", dtype=torch.float32, requires_grad=True)
            else:
                tex_img = torch.ones([768, 768, 3], device="cuda", requires_grad=True)
            optimizer = torch.optim.Adam([tex_img], lr=0.01)
            tex_steps = args.texture_steps
            for it in tqdm(range(tex_steps)):
                optimizer.zero_grad()

                rn_imgs, _, _ = renderer.render(v, n, f, tex_img, vt, ft)
                batch_size = rn_imgs.size(0)
                if args.vis and it %10 ==0:           
                    # tex_vis = to_np(tex_img.clone())
                    num_cycles = 5
                    it_vis = it/10
                    t = (it_vis / tex_steps) * num_cycles
                    view_idx = int((1 - abs(2 * (t % 1) - 1)) * (num_imgs - 1))
                    rn_vis = to_np(rn_imgs[view_idx, :,:, :3])
                    rn_vis = np.clip(rn_vis, 0, 1)
                    gt_vis = to_np(ref_imgs[view_idx, :, :, :3])
                    gt_vis = np.clip(gt_vis, 0, 1)
                    canvas = np.concatenate([rn_vis, gt_vis], axis=1)
                    canvas = (255*canvas).astype(np.uint8)
                    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)            
                    cv2.imshow("vis", canvas)
                    # cv2.imshow("vis", tex_vis)
                    cv2.waitKey(1)

                loss = F.l1_loss(rn_imgs[..., :3], ref_imgs[..., :3])
                # loss +=1e-3 * total_variation_loss(tex_img.permute(2, 0, 1))
                loss.backward()
                optimizer.step()

            tex_vis = to_np(tex_img)
            tex_vis = np.clip(tex_vis, 0, 1)
            tex_vis = (255*tex_vis).astype(np.uint8)
            tex_vis = cv2.cvtColor(tex_vis, cv2.COLOR_RGB2BGR)

            if args.save:
                tex_path = os.path.join(save_dir, "flame_texture_fine.jpg")
                cv2.imwrite(tex_path, tex_vis)

            # pt3ds = position_map[position_mask].detach().cpu().numpy()
            # colors = tex_img[position_mask].detach().cpu().numpy()
            # pcd = o3d.geometry.PointCloud()
            # pcd.points=o3d.utility.Vector3dVector(pt3ds.reshape(-1, 3))
            # pcd.colors=o3d.utility.Vector3dVector(colors.reshape(-1, 3))
            # mesh = o3d.io.read_triangle_mesh(os.path.join(save_dir, "flame_nonrigid_mesh_wo_eye.ply"))
            # mesh.compute_vertex_normals()
            # o3d.visualization.draw_geometries([pcd, mesh])

        else:
            continue
            
        # except Exception as e:
        #     print(e)
        #     continue    
        