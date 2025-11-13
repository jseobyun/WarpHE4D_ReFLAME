import os
import cv2
import math
import torch
import pickle
import trimesh
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize
from ext.dwpose import DwposeDetector
from pytorch3d.io import IO
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes

def save_flame_params(save_path, flame_params):
    shape_param = flame_params[0][0].reshape(-1) # 300
    expr_param = flame_params[1][0].reshape(-1) # 100
    pose_param = flame_params[2][0].reshape(-1)
    s = flame_params[3][0].reshape(-1)
    t = flame_params[4][0].reshape(-1)
    save_params = torch.cat([shape_param, expr_param, pose_param, s, t]).detach().cpu().numpy()    
    np.save(save_path, save_params)

def aa2matrix(aa6):
    """
    Convert axis-angle (3 or 6-dim) to rotation matrix (3x3)
    Supports both numpy and torch tensors.
    """
    if isinstance(aa6, torch.Tensor):
        w = aa6[..., :3].detach().cpu().numpy()
        R, _ = cv2.Rodrigues(w)
        R = torch.from_numpy(R).to(aa6.device, dtype=aa6.dtype)
    else:
        w = np.asarray(aa6, dtype=np.float64)[..., :3]
        R, _ = cv2.Rodrigues(w)
    return R


def matrix2aa(R):
    """
    Convert rotation matrix (3x3) to axis-angle (3,)
    Supports both numpy and torch tensors.
    """
    if isinstance(R, torch.Tensor):
        R_np = R.detach().cpu().numpy()
        aa, _ = cv2.Rodrigues(R_np)
        aa = torch.from_numpy(aa.squeeze()).to(R.device, dtype=R.dtype)
    else:
        R_np = np.asarray(R, dtype=np.float64)
        aa, _ = cv2.Rodrigues(R_np)
        aa = aa.squeeze()
    return aa

def fit_sphere_with_init(l_points, r_points, l_init_center, r_init_center):
    """
    points: (N, 3) numpy array
    init_center: (3,) numpy array
    returns: center (3,), radius (float)
    """

    def objective(params):
        lcx, lcy, lcz, rcx, rcy, rcz, r = params
        l_dists = np.linalg.norm(l_points - np.array([lcx, lcy, lcz]).reshape(1, 3), axis=1)
        r_dists = np.linalg.norm(r_points - np.array([rcx, rcy, rcz]).reshape(1, 3), axis=1)

        l_dist = np.mean((l_dists - r) ** 2)
        r_dist = np.mean((r_dists - r) ** 2)
        push_dist = max((2 * r) ** 2 - ((lcx - rcx) ** 2 + (lcy - rcy) ** 2 + (lcz - rcz) ** 2), 0)
        radius_reg = max(r / 0.1301791 - 1, 0)
        return l_dist + r_dist + push_dist + 0.1 * radius_reg

    # 초기 반지름: 중심으로부터 평균 거리
    init_r = np.mean(np.linalg.norm(l_points - l_init_center, axis=1))

    x0 = np.concatenate([l_init_center, r_init_center, [init_r]])

    result = minimize(objective, x0, method='L-BFGS-B')
    lcx, lcy, lcz, rcx, rcy, rcz, r = result.x
    return np.array([lcx, lcy, lcz]), np.array([rcx, rcy, rcz]), r


def optimize_eyeballs(v, lmk, v_init):
    l_init_offset = v_init[3931:4477] - np.mean(v_init[3931:4477], axis=0).reshape(1, 3)
    r_init_offset = v_init[4477:] - np.mean(v_init[4477:], axis=0).reshape(1, 3)
    l_init_raidus = np.mean(
        np.sqrt(np.sum((v_init[3931:4477] - np.mean(v_init[3931:4477], axis=0).reshape(1, 3)) ** 2, axis=-1)))
    r_init_raidus = np.mean(
        np.sqrt(np.sum((v_init[4477:] - np.mean(v_init[4477:], axis=0).reshape(1, 3)) ** 2, axis=-1)))
    l_init_center = v[3929]
    r_init_center = v[3930]

    l_init_points = lmk[42:48]
    r_init_points = lmk[36:42]

    l_center, r_center, lr_radius = fit_sphere_with_init(l_init_points, r_init_points, l_init_center, r_init_center)

    l_points = l_init_offset * (lr_radius / l_init_raidus) + l_center.reshape(1, 3)
    r_points = r_init_offset * (lr_radius / r_init_raidus) + r_center.reshape(1, 3)

    v[3929] = l_center
    v[3930] = r_center
    v[3931:4477] = l_points
    v[4477:] = r_points
    return v


def to_np(a):
    return a.detach().cpu().numpy()


def to_cuda(a):
    return torch.from_numpy(a).cuda()


def load_mesh_under_dir(data_dir):
    '''
    Let assume there is only one obj file in the data_dir
    '''

    file_names = sorted(os.listdir(data_dir))
    obj_name = None
    material_name = None
    for file_name in file_names:        
        if file_name.endswith(".mtl"):
            material_name = file_name
            obj_name = material_name.replace(".mtl", ".obj")    


    # if obj_name is None or material_name is None:
    #     raise Exception(f"{data_dir} does not contain a valid obj file.")
    
    
    texture_name = None
    if material_name is not None:
        mtl_path = os.path.join(data_dir, material_name)
        with open(mtl_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('map_Kd'):
                    # shlex handles quoted strings and spaces
                    tokens = line.split(" ")
                    if len(tokens) >= 2:
                        texture_name = tokens[-1]

    # if texture_name is None:
    #     raise Exception(f"{data_dir} does not contain a valid obj file.")
    mesh_path = None
    if obj_name is not None:
        mesh_path = os.path.join(data_dir, obj_name)

    if obj_name is None or not os.path.exists(mesh_path):
        mesh_path = os.path.join(data_dir, "raw.obj")
    if obj_name is None or not os.path.exists(mesh_path): # th rp
        obj_name = os.path.basename(data_dir)
        mesh_path = os.path.join(data_dir, obj_name+".obj")

    if texture_name is not None:
        texture_path = os.path.join(data_dir, texture_name)
    else:
        texture_path = os.path.join(data_dir, obj_name+".jpg")
    mesh = trimesh.load(mesh_path, process=False)
    tex_img = Image.open(texture_path).resize((2048, 2048))  # 2k texture map for memory saving
    tex_img = np.asarray(tex_img).astype(np.float32) / 255.0

    verts = np.asarray(mesh.vertices).astype(np.float32)    

    mesh.vertices = verts

    faces = np.asarray(mesh.faces).astype(np.int32)
    normals = np.asarray(mesh.vertex_normals).astype(np.float32)
    uvs = np.asarray(mesh.visual.uv).astype(np.float32)
    uvs[:, 1] = 1 - uvs[:, 1]

    v = torch.from_numpy(verts).cuda()
    n = torch.from_numpy(normals).cuda()
    f = torch.from_numpy(faces).cuda()
    uv = torch.from_numpy(uvs).cuda()
    tex_img = torch.from_numpy(tex_img).cuda()
    return v, n, f, uv, tex_img



def load_mesh_rn360(data_dir):
    mesh_path = os.path.join(data_dir, "mesh.obj")
    M = np.load(os.path.join(data_dir, "mesh_coord_changer.npy"))
    texture_path = os.path.join(data_dir, "mesh.jpg")
    Rx = np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]], dtype=np.float32)

    mesh = trimesh.load(mesh_path, process=False)
    tex_img = Image.open(texture_path).resize((2048, 2048))  # 2k texture map for memory saving
    tex_img = np.asarray(tex_img).astype(np.float32) / 255.0

    verts = np.asarray(mesh.vertices).astype(np.float32)    

    
    verts = (M[:3, :3]@verts.transpose(1,0)).transpose(1,0) + M[:3,-1].reshape(1, 3)
    verts = (Rx@verts.transpose(1,0)).transpose(1,0)
    verts[:,1] -= 0.45
    verts *= 7
    mesh.vertices = verts

    faces = np.asarray(mesh.faces).astype(np.int32)
    normals = np.asarray(mesh.vertex_normals).astype(np.float32)
    uvs = np.asarray(mesh.visual.uv).astype(np.float32)
    uvs[:, 1] = 1 - uvs[:, 1]

    v = torch.from_numpy(verts).cuda()
    n = torch.from_numpy(normals).cuda()
    f = torch.from_numpy(faces).cuda()
    uv = torch.from_numpy(uvs).cuda()
    tex_img = torch.from_numpy(tex_img).cuda()
    return v, n, f, uv, tex_img



def total_variation_loss(img):
    return torch.mean(torch.abs(img[:, :, :-1] - img[:, :, 1:])) + \
        torch.mean(torch.abs(img[:, :-1, :] - img[:, 1:, :]))


def gen_orbit_views(elev_range=0, azim_range=80, front_range=30, num_gens=40, dist=4.3):    
    elev = 0
    if elev_range != 0:
        elevs = np.linspace(-abs(elev_range), abs(elev_range), num_gens) / 180 * np.pi
    else:
        elevs = [0]

    azims = np.linspace(-abs(azim_range), abs(azim_range), num_gens) / 180 * np.pi

    T_kgs = []
    T_gks = []
    view_mask = []
    for elev in elevs:
        for azim in azims:
            x = dist * np.cos(elev) * np.sin(azim)
            y = dist * np.sin(elev)
            z = dist * np.cos(elev) * np.cos(azim)

            center = np.array([x, y, z])

            upvector = np.array([0.0, 1.0, 0.0])
            zaxis = -center / np.linalg.norm(center)
            xaxis = np.cross(zaxis, upvector)
            xaxis = xaxis / np.linalg.norm(xaxis)
            yaxis = np.cross(zaxis, xaxis)
            yaxis = yaxis / np.linalg.norm(yaxis)

            R = np.concatenate([xaxis[:, None], yaxis[:, None], zaxis[:, None]], axis=1)  # T_gk
            t = center
            T_gk = np.eye(4).astype(np.float32)
            T_gk[:3, :3] = R
            T_gk[:3, -1] = t
            T_gk = torch.from_numpy(T_gk).cuda()
            T_kg = torch.linalg.inv(T_gk)
            T_kgs.append(T_kg)
            T_gks.append(T_gk)
            if azim >= -abs(front_range) / 180 * np.pi and azim <= abs(front_range) / 180 * np.pi:
                view_mask.append(True)  # valid almost front view
            else:
                view_mask.append(False)
    return T_kgs, T_gks, np.asarray(view_mask)


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    vertices = vertices.reshape(1, -1, 3)
    faces = faces.long()
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])

    ear_and_eye = torch.tensor([4597, 4051, 26, 162]).long().to(device=landmarks.device)
    extra_landmarks = vertices[:, ear_and_eye]

    landmarks = torch.cat([landmarks, extra_landmarks], dim=1)
    return landmarks[0]


def load_flame_meta(template_dir):
    flame_lmk_embedding_path = os.path.join(template_dir, "landmark_embedding.npy")
    lmk_embeddings = np.load(flame_lmk_embedding_path, allow_pickle=True, encoding='latin1')
    lmk_embeddings = lmk_embeddings[()]
    full_lmk_faces_idx = torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long().cuda()
    full_lmk_bary_coords = torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(torch.float32).cuda()
    template_mesh = np.load(os.path.join(template_dir, "flame_template.npz"))
    return template_mesh, full_lmk_faces_idx, full_lmk_bary_coords


def load_flame_uv(template_dir):
    template_mesh = np.load(os.path.join(template_dir, "flame_template.npz"))
    f = np.asarray(template_mesh["faces"]).astype(np.int32)
    vt = np.asarray(template_mesh["uv"]).astype(np.float32)
    ft = np.asarray(template_mesh["uv_faces"]).astype(np.int32)
    vt = 1 - vt
    return vt, ft, f


def build_o3d_mesh(v, f, color=None):
    v = v.reshape(-1, 3)
    f = f.reshape(-1, 3)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(v),
        triangles=o3d.utility.Vector3iVector(f),
    )
    if color is not None:
        vc = np.zeros_like(v)
        if color == "r":
            vc[:, 0] = 1.0
            vc[:, 1] = 0.533
            vc[:, 2] = 0.0
        elif color == "g":
            vc[:, 0] = 0.6
            vc[:, 1] = 1.0
            vc[:, 2] = 0.6
        elif color == "b":
            vc[:, 2] = 0.5
        mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
    mesh.compute_vertex_normals()
    return mesh

def build_o3d_pcd(points, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    if color is not None:
        vc = np.zeros_like(points.reshape(-1, 3))
        if color == "r":
            vc[:, 0] = 1.0
            vc[:, 1] = 0.533
            vc[:, 2] = 0.0
        elif color == "g":
            vc[:, 0] = 0.6
            vc[:, 1] = 1.0
            vc[:, 2] = 0.6
        elif color == "b":
            vc[:, 2] = 0.5
        pcd.colors = o3d.utility.Vector3dVector(vc)
    return pcd

def backproject_depth(K, depth_rn, color="g"):    
    img_h, img_w = np.shape(depth_rn)[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(0, img_w), np.arange(0, img_h), indexing="xy")
    u = (u - cx) / fx
    v = (v - cy) / fy

    xy1 = np.concatenate([u[:,:,None], v[:,:,None], np.ones_like(u[:,:,None])], axis=2) # HW3
    d = depth_rn.reshape(img_h, img_w, 1)

    xyz = xy1 * d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
    vc = np.zeros_like(xyz.reshape(-1, 3))
    if color == "r":
        vc[:, 0] = 1.0
        vc[:, 1] = 0.533
        vc[:, 2] = 0.0
    elif color == "g":
        vc[:, 0] = 0.6
        vc[:, 1] = 1.0
        vc[:, 2] = 0.6
    elif color == "b":
        vc[:, 2] = 0.5
    pcd.colors = o3d.utility.Vector3dVector(vc)
    
    return pcd    


def apply_T(T, points):
    if len(T.shape) != 2 or len(points.shape) != 2:
        raise Exception("ERROR : the dimensions of transformation matrix and points are wrong.")

    points_ = np.matmul(T[:3, :3], points.transpose(1, 0)).transpose(1, 0) + T[:3, -1].reshape(-1, 3)
    return points_

def make_origin(T_gk=np.eye(4), scale=1):
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]) * scale

    points = apply_T(T_gk, points)
    origin_line = [[0, 1], [0, 2], [0, 3]]
    origin_color = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    origin = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(origin_line),
    )
    origin.colors = o3d.utility.Vector3dVector(origin_color)
    return origin


#########################


def common_annotator_call(model, tensor_image, input_batch=False, show_pbar=True, **kwargs):
    if "detect_resolution" in kwargs:
        del kwargs["detect_resolution"]  # Prevent weird case?

    if "resolution" in kwargs:
        detect_resolution = kwargs["resolution"] if type(kwargs["resolution"]) == int and kwargs[
            "resolution"] >= 64 else 512
        del kwargs["resolution"]
    else:
        detect_resolution = 512

    if input_batch:
        np_images = np.asarray(tensor_image * 255., dtype=np.uint8)
        np_results = model(np_images, output_type="np", detect_resolution=detect_resolution, **kwargs)
        return torch.from_numpy(np_results.astype(np.float32) / 255.0)

    batch_size = tensor_image.shape[0]

    out_tensor = None
    for i, image in enumerate(tensor_image):
        np_image = np.asarray(image.cpu() * 255., dtype=np.uint8)
        np_result = model(np_image, output_type="np", detect_resolution=detect_resolution, **kwargs)
        out = torch.from_numpy(np_result.astype(np.float32) / 255.0)
        if out_tensor is None:
            out_tensor = torch.zeros(batch_size, *out.shape, dtype=torch.float32)
        out_tensor[i] = out
    return out_tensor

def load_p3d_mesh(mesh_path, subdiv=0, device="cuda"):
    
    mesh = IO().load_mesh(mesh_path, device=device)

    if subdiv !=0:
        subdivider = SubdivideMeshes()
        for iter in range(subdiv):
            mesh = subdivider(mesh)
    return mesh    

def subdiv_uv(uvs):
    num_faces = np.shape(uvs)[0]
    new_uvs = []
    for fidx, face_uv in enumerate(uvs):
        v0 = face_uv[0]
        v1 = face_uv[1]
        v2 = face_uv[2]
        v3 = (v0 + v1) / 2
        v4 = (v0 + v2) / 2
        v5 = (v1 + v2) / 2

        f0 = np.vstack([v0, v3, v4])
        f1 = np.vstack([v1, v5, v3])
        f2 = np.vstack([v2, v4, v5])
        f3 = np.vstack([v5, v4, v3])

        subdiv_uvs = []

        subdiv_uvs.append(f0.reshape(-1, 3, 2))
        subdiv_uvs.append(f1.reshape(-1, 3, 2))
        subdiv_uvs.append(f2.reshape(-1, 3, 2))
        subdiv_uvs.append(f3.reshape(-1, 3, 2))
        subdiv_uvs = np.concatenate(subdiv_uvs, axis=0) # 4 3 2
        new_uvs.append(subdiv_uvs.reshape(1, 4, 3, 2))

        '''    
            e.g. subdivided face
            ::
                       v0
                       /\
                      /  \
                     / f0 \
                 v4 /______\ v3
                   /\      /\
                  /  \ f3 /  \
                 / f2 \  / f1 \
                /______\/______\
               v2       v5       v1

               f0 = [0, 3, 4]
               f1 = [1, 5, 3]
               f2 = [2, 4, 5]
               f3 = [5, 4, 3]
        '''
        # Map from packed faces to packed edges. This represents the index of
        # the edge opposite the vertex for each vertex in the face. E.g.
        #
        #         v0
        #         /\
        #        /  \
        #    e1 /    \ e2
        #      /      \
        #     /________\
        #   v2    e0   v1
        #
        # Face (v0, v1, v2) => Edges (e0, e1, e2)
    new_uvs = np.concatenate(new_uvs, axis=0).transpose(1,0, 2, 3) # N 4 3 2
    new_uvs = new_uvs.reshape(-1, 3, 2)
    assert np.shape(new_uvs)[0] == num_faces * 4
    return new_uvs


class FaceKeypointDetector():
    def __init__(self):

        ### dwpose
        bbox_detector = "yolox_l.onnx"
        pose_estimator = "dw-ll_ucoco_384.onnx"
        if bbox_detector == "yolox_l.onnx":
            yolo_repo = "yzd-v/DWPose"
        elif "yolox" in bbox_detector:
            yolo_repo = "hr16/yolox-onnx"
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
        else:
            raise NotImplementedError(f"Download mechanism for {bbox_detector}")

        if pose_estimator == "dw-ll_ucoco_384.onnx":
            pose_repo = "yzd-v/DWPose"
        elif pose_estimator.endswith(".onnx"):
            pose_repo = "hr16/UnJIT-DWPose"
        elif pose_estimator.endswith(".torchscript.pt"):
            pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
        else:
            raise NotImplementedError(f"Download mechanism for {pose_estimator}")

        self.dwpose = DwposeDetector.from_pretrained(
            pose_repo,
            yolo_repo,
            det_filename=bbox_detector, pose_filename=pose_estimator,
            torchscript_device=torch.device("cuda"),
        )

    def estimate_pose(self, image, detect_hand="disable", detect_body="disable", detect_face="enable", resolution=512,

                      scale_stick_for_xinsr_cn="disable", **kwargs):

        detect_hand = detect_hand == "enable"
        detect_body = detect_body == "enable"
        detect_face = detect_face == "enable"
        scale_stick_for_xinsr_cn = scale_stick_for_xinsr_cn == "enable"
        openpose_dicts = []

        def func(image, **kwargs):
            pose_img, openpose_dict = self.dwpose(image, **kwargs)
            openpose_dicts.append(openpose_dict)
            return pose_img

        out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face,
                                    include_body=detect_body, image_and_json=True, resolution=resolution,
                                    xinsr_stick_scaling=scale_stick_for_xinsr_cn)
        return openpose_dicts
