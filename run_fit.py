import os
import cv2
import torch
import trimesh
import argparse
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from PIL import Image
from tqdm import trange, tqdm
from largesteps.load_xml import load_scene
from largesteps.render import NVDRenderer
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix, compute_vertex_normals, compute_face_normals
from largesteps.optimize import AdamUniform
from warphead.utils import (optimize_eyeballs, to_np, load_mesh_under_dir,
                            total_variation_loss, gen_orbit_views, vertices2landmarks,
                            load_flame_meta, to_cuda, build_o3d_mesh, FaceKeypointDetector)



def parse_config():
    parser = argparse.ArgumentParser("ReFLAME configuration")
    ### mandatory
    parser.add_argument("--data_dir", type=str, default="/home/jseob/Downloads/uv_fit_test/013")
    parser.add_argument("--opt_lambda", type=int, default=30, help="lambda used in optimization")
    parser.add_argument("--coarse_steps", type=int, default=700, help="Coarse fitting iterations")
    parser.add_argument("--refine_steps", type=int, default=300, help="Refine fitting iterations")
    parser.add_argument("--texture_steps", type=int, default=1000, help="1K texture map optimization iterations")
    parser.add_argument("--azim_range", type=float, default=150, help="azimuth angle range [-a, a] for renderer. 150 degree recomennded")

    ### optional : change these if you need.
    parser.add_argument("--w_uv", type=float, default=1.0, help="weight of UV loss (Coarse + Refine)")
    parser.add_argument("--w_mask", type=float, default=5.0, help="weight of mask loss (Coarse)")
    parser.add_argument("--w_depth_c", type=float, default=0.5, help="weight of depth loss (Coarse)")
    parser.add_argument("--w_kp_c", type=float, default=0.005, help="weight of depth loss (Coarse)")
    parser.add_argument("--w_depth_r", type=float, default=1.0, help="weight of depth loss (Refine)")
    parser.add_argument("--w_kp_r", type=float, default=1.0, help="weight of depth loss (Refine)")

    ### debugging
    parser.add_argument("--vis", action="store_true", default=True, help="enable visualization for debugging")
    parser.add_argument("--save", action="store_true", default=False, help="enable save")
    args = parser.parse_args()
    if args.save:
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./results/images", exist_ok=True)
        os.makedirs("./results/meshes", exist_ok=True)
    return args

if __name__ == "__main__":

    args = parse_config()
    subj_name = args.data_dir.split("/")[-1]
    '''
    Load pretrained models
    '''
    kp_detector = FaceKeypointDetector()
    warphead = torch.load("./ckpts/warphead/warph3ad_fast.pt", map_location="cuda", weights_only=False)

    '''
    Load renderer settings : envmap is only for visualization.
    '''

    filepath = "./templates/ENV/environment.xml"
    scene_params = load_scene(filepath)
    T_kgs, T_gks, view_mask = gen_orbit_views(azim_range=args.azim_range)
    scene_params["view_mats"] = T_kgs

    T_kgs = torch.stack(T_kgs, dim=0)
    T_gks = torch.stack(T_gks, dim=0)

    renderer = NVDRenderer(scene_params, shading=False, boost=3) # check shading False during optimization.

    '''
    Load reference (target) mesh and FLAME meta info.
    '''
    v_ref, n_ref, f_ref, uv_ref, tex_img = load_mesh_under_dir(args.data_dir)
    template_mesh, full_lmk_faces_idx, full_lmk_bary_coords = load_flame_meta("./templates/FLAME2023")

    v = np.asarray(template_mesh["vertices"]).astype(np.float32)
    f = np.asarray(template_mesh["faces"]).astype(np.int32)
    vt = np.asarray(template_mesh["uv"]).astype(np.float32)
    ft = np.asarray(template_mesh["uv_faces"]).astype(np.int32)
    vt = 1 - vt  # essential for matching coordinates.

    v = to_cuda(v)
    f = to_cuda(f)
    vt = to_cuda(vt)
    ft = to_cuda(ft)

    '''
    Render reference images, masks, and depths
    '''
    ref_imgs, ref_masks, ref_depths = renderer.render(v_ref, n_ref, f_ref, tex_img, uv_ref)
    ref_uvs = []
    valid_masks = []
    lmk_uv = []
    bary_coords = full_lmk_bary_coords[0]
    face_idxs = full_lmk_faces_idx[0]

    for i, face_idx in enumerate(face_idxs): # calculate uv coordinates of landmarks
        a, b, c = ft[face_idx]
        uv = vt[a] * bary_coords[i][0] + vt[b] * bary_coords[i][1] + vt[c] * bary_coords[i][2]
        lmk_uv.append(uv.reshape(1, 2))
    lmk_uv = torch.cat(lmk_uv, dim=0)

    num_imgs = ref_imgs.size(0)

    '''
    Extract face landmarks (68 + 2 eye centers + 2 ear centers from COCO body keypoints)
    
    Thank you DWPose!
    '''
    flipped_ref_imgs = torch.flip(ref_imgs, dims=[2]) # ugly trick for matching coordinates.

    start, end = np.flatnonzero(view_mask)[[0, -1]]

    kps = kp_detector.estimate_pose(flipped_ref_imgs[start:end])
    kp_points = np.zeros([num_imgs, 72, 3], dtype=np.float32)
    for idx in range(end-start) :
        try:
            kp = kps[idx]
            kp_coords = kp["people"][0]["face_keypoints_2d"]

            img_w = kp["canvas_width"]
            img_h = kp["canvas_height"]

            ### set weight per landmark. tune this if you need.
            kp_coords = np.asarray(kp_coords).reshape(-1, 3) # 70, 3
            kp_coords[:, 0] = img_w - kp_coords[:, 0]
            kp_coords[:, 2] = 0.0
            kp_coords[:17, 2] = 0.0 # contour
            kp_coords[17:22, 2] = 0.0 # right eyebrow
            kp_coords[22:27, 2] = 0.0 # left_eyebrow
            kp_coords[36:42, 2] = 5.0  # right eye
            kp_coords[42:48, 2] = 5.0  # left eye
            kp_coords[48:, 2] = 2.0  # lips
            kp_coords[31:36, 2] = 0.0  # under nose
            kp_coords[68:70, 2] = 0.0 # eye center. it is for eyeballs but not that effective.

            body_coords = kp["people"][0]["pose_keypoints_2d"]
            body_coords = np.asarray(body_coords).reshape(-1, 3) # 18 3 , Rear, Lear
            body_coords[:, -1] = 0.0 # ears. this may cause protruding ears.
            body_coords = body_coords[::-1] # flip for matching the order.

            kp_coords = np.concatenate([kp_coords, body_coords[:2, :]], axis=0)
            kp_points[start+idx] = kp_coords.reshape(-1, 3)
        except:
            kp_points[start+idx] = np.zeros([1, 72, 3], dtype=np.float32)


    kp_points = to_cuda(kp_points)
    batch_size = kp_points.size(0)
    view_mask = to_cuda(view_mask)
    kp_valids = torch.sum(kp_points.reshape(batch_size, -1), dim=-1) != 0
    kp_valids = kp_valids * view_mask # all fails + out of front view angles

    '''
    Extract Head UV maps (UV mapping from HIFI3D FLAME)
    
    UV extractor is my WarpHEAD re-trained only for UV (except Depth). 
    '''
    for i in range(ref_imgs.size(0)):
        ref_img = to_np(ref_imgs[i, :, :, :3])
        ref_img = np.clip(ref_img, 0, 1)
        ref_img = (255*ref_img).astype(np.uint8)
        ref_img_pil = Image.fromarray(ref_img)
        ref_img_cv = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            warp, conf = warphead.inference(ref_img_pil)
            uv = (warp + 1)/2 # (-1, 1) to (0, 1)
            conf = conf.clone()
            ref_mask = ref_masks[i, :,:,0]
            valid_mask = (ref_mask != 0) * (conf >=0.01) # trust low confidence region together. it is still good!
            conf[valid_mask] = 1.0
            conf[~valid_mask] = 0.0

        uvc = torch.cat([uv, conf[:, :, None]], dim=-1).unsqueeze(dim=0)
        valid_mask = valid_mask.unsqueeze(dim=0)

        ref_uvs.append(uvc)
        valid_masks.append(valid_mask)

    ref_uvs = torch.cat(ref_uvs, dim=0)
    valid_masks = torch.cat(valid_masks, dim=0)

    '''
    Setting optimization
    
    Thank you large-steps!
    '''
    steps = args.coarse_steps + args.refine_steps
    coarse_steps = 1000# Number of optimization steps
    refine_steps = args.refine_steps
    step_size = 0.03 #3e-2

    flame_texture = np.ones([1024, 1024, 3], dtype=np.float32)
    us, vs = np.meshgrid(np.arange(0, 1024), np.arange(0, 1024), indexing="xy")
    us = us / 1024
    vs = vs / 1024
    flame_texture[:, :, 0] = us
    flame_texture[:, :, 1] = vs
    flame_tex = to_cuda(flame_texture)

    v_init = to_np(v) # it is for eyeballs postprocessing at the end of the optimization.
    lambda_ = args.opt_lambda# Hyperparameter lambda of our method, used to compute the matrix (I + lambda_*L)
    M = compute_matrix(v, f, lambda_)

    # Parameterize
    u = to_differential(M, v)
    u.requires_grad = True

    opt = AdamUniform([u], step_size)

    '''
    Main optimization
    '''
    for it in trange(steps):
        v = from_differential(M, u, 'Cholesky')
        face_normals = compute_face_normals(v, f)
        n = compute_vertex_normals(v, f, face_normals)

        opt_imgs, opt_masks, opt_depths = renderer.render(v, n, f, flame_tex, vt, ft)
        if args.vis and it %10 ==0:
            save_geom_dir = "./results/images/geom"
            os.makedirs(save_geom_dir, exist_ok=True)

            renderer.shading = True # turn on shading for better visualization
            vis_imgs, _, _= renderer.render(v, n, f, flame_tex, vt, ft)
            num_cycles = 5
            it_vis = it/10
            t = (it_vis / steps) * num_cycles
            view_idx = int((1 - abs(2 * (t % 1) - 1)) * (num_imgs - 1))


            vis_img = to_np(vis_imgs[view_idx, :, :, :3])
            ref_uv = to_np(ref_uvs[view_idx, :, :, :3])
            ref_uv[ref_uv[:,:,-1] ==0] =0.0
            ref_img = to_np(ref_imgs[view_idx, :, :, :3])
            overlay = cv2.addWeighted(ref_img, 0.5, ref_uv, 0.5, 0)

            canvas = np.concatenate([overlay, vis_img], axis=1)
            canvas = np.clip(canvas, 0, 1)
            canvas = (255*canvas).astype(np.uint8)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            if args.save:
                cv2.imwrite(os.path.join(save_geom_dir, "geom_"+format(it, "06d")+".jpg"), canvas)
            cv2.imshow("coarse", canvas)
            cv2.waitKey(1)
            renderer.shading = False # turn off again for accururate optimization

        ### Compute landmark loss
        lmk_curr = vertices2landmarks(v, f, full_lmk_faces_idx, full_lmk_bary_coords)
        lmk_i = renderer.project(lmk_curr) * 448
        loss_kp = (lmk_i[kp_valids] - kp_points[kp_valids][..., :2]) * (kp_points[kp_valids][..., 2:])
        loss_kp = loss_kp.abs().mean()

        ### Compute depth loss
        valid = (ref_depths != 0) * (opt_depths != 0) # strict region control
        loss_depth = (ref_depths[valid] - opt_depths[valid]).abs().mean()

        ### Compute mask loss (silhouette loss)
        loss_mask = (opt_masks - ref_masks)
        loss_mask = loss_mask.abs().mean()

        ### Compute UV loss (main!)
        opt_valid = opt_imgs[valid_masks][..., :2]
        ref_valid = ref_uvs[valid_masks][..., :2]

        loss_2d = (opt_valid - ref_valid)
        loss_2d = loss_2d.abs().mean()

        if it <= coarse_steps:
            loss = args.w_uv * loss_2d + args.w_mask * loss_mask + args.w_kp_c * loss_kp + args.w_depth_c * loss_depth
        else:
            loss = args.w_uv * loss_2d + args.w_kp_r * loss_kp + args.w_depth_r * loss_depth


        # Backpropagate
        opt.zero_grad()
        loss.backward()

        # Update parameters
        opt.step()

    '''
    Postprocess eyeballs
    '''
    lmk = vertices2landmarks(v, f, full_lmk_faces_idx, full_lmk_bary_coords)
    lmk = to_np(lmk)
    v = to_np(v)
    f = to_np(f)
    v = optimize_eyeballs(v, lmk, v_init)

    v_final = to_cuda(v)
    f_final = to_cuda(f)
    face_normals = compute_face_normals(v_final, f_final)
    n_final = compute_vertex_normals(v_final, f_final, face_normals)

    '''
    Texture map optimization
    '''
    tex_img = torch.ones([1024, 1024, 3], device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam([tex_img], lr=0.005)
    tex_steps = args.texture_steps
    for it in tqdm(range(tex_steps)):
        optimizer.zero_grad()

        rn_imgs, _, _ = renderer.render(v_final, n_final, f_final, tex_img, vt, ft)
        batch_size = rn_imgs.size(0)

        if args.vis and it %10 ==0:
            save_tex_dir = "./results/images/tex"
            os.makedirs(save_tex_dir, exist_ok=True)

            num_cycles = 5
            it_vis = it/10
            t = (it_vis / steps) * num_cycles
            view_idx = int((1 - abs(2 * (t % 1) - 1)) * (num_imgs - 1))
            rn_vis = to_np(rn_imgs[view_idx, :,:, :3])
            rn_vis = np.clip(rn_vis, 0, 1)
            gt_vis = to_np(ref_imgs[view_idx, :, :, :3])
            gt_vis = np.clip(gt_vis, 0, 1)
            canvas = np.concatenate([rn_vis, gt_vis], axis=1)
            canvas = (255*canvas).astype(np.uint8)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            if args.save:
                cv2.imwrite(os.path.join(save_tex_dir, "tex_" + format(it, "06d") + ".jpg"), canvas)
            cv2.imshow("vis", canvas)
            cv2.waitKey(1)
        loss = F.l1_loss(rn_imgs[..., :3], ref_imgs[..., :3])
        loss +=1e-4 * total_variation_loss(tex_img.permute(2, 0, 1))
        loss.backward()
        optimizer.step()

    tex_vis = to_np(tex_img)
    tex_vis = np.clip(tex_vis, 0, 1)
    tex_vis = (255*tex_vis).astype(np.uint8)
    tex_vis = cv2.cvtColor(tex_vis, cv2.COLOR_RGB2BGR)

    if args.vis:
        cv2.imwrite("./results/meshes/texturemap.jpg", tex_vis)
        cv2.imshow("tex", tex_vis)
        cv2.waitKey(0)


        final_mesh = build_o3d_mesh(v, f, color="r")
        raw_mesh = build_o3d_mesh(to_np(v_ref), to_np(f_ref), color="g")

        o3d.visualization.draw_geometries([final_mesh])
        o3d.visualization.draw_geometries([raw_mesh, final_mesh])
