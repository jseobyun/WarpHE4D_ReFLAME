import os
import math
import torch
import trimesh
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize
from ext.dwpose import DwposeDetector

@torch.no_grad()
def cls_to_flow_refine(cls):
    B, C, H, W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(
        *[torch.linspace(-1 + 1 / res, 1 - 1 / res, steps=res, device=device) for _ in range(2)],
        indexing='ij'
    )
    G = torch.stack([G[1], G[0]], dim=-1).reshape(C, 2)
    # FIXME: below softmax line causes mps to bug, don't know why.
    if device.type == 'mps':
        cls = cls.log_softmax(dim=1).exp()
    else:
        cls = cls.softmax(dim=1)
    mode = cls.max(dim=1).indices

    index = torch.stack((mode - 1, mode, mode + 1, mode - res, mode + res), dim=1).clamp(0, C - 1).long()
    neighbours = torch.gather(cls, dim=1, index=index)[..., None]
    flow = neighbours[:, 0] * G[index[:, 0]] + neighbours[:, 1] * G[index[:, 1]] + neighbours[:, 2] * G[
        index[:, 2]] + neighbours[:, 3] * G[index[:, 3]] + neighbours[:, 4] * G[index[:, 4]]
    tot_prob = neighbours.sum(dim=1)
    flow = flow / tot_prob
    return flow


def local_correlation(
        feature0,
        feature1,
        local_radius,
        padding_mode="zeros",
        flow=None,
        sample_mode="bilinear",
):
    r = local_radius
    K = (2 * r + 1) ** 2
    B, c, h, w = feature0.size()
    corr = torch.empty((B, K, h, w), device=feature0.device, dtype=feature0.dtype)
    if flow is None:
        # If flow is None, assume feature0 and feature1 are aligned
        coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=feature0.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=feature0.device),
            ),
            indexing='ij'
        )
        coords = torch.stack((coords[1], coords[0]), dim=-1)[
            None
        ].expand(B, h, w, 2)
    else:
        coords = flow.permute(0, 2, 3, 1)  # If using flow, sample around flow target.
    local_window = torch.meshgrid(
        (
            torch.linspace(-2 * local_radius / h, 2 * local_radius / h, 2 * r + 1, device=feature0.device),
            torch.linspace(-2 * local_radius / w, 2 * local_radius / w, 2 * r + 1, device=feature0.device),
        ),
        indexing='ij'
    )
    local_window = torch.stack((local_window[1], local_window[0]), dim=-1)[
        None
    ].expand(1, 2 * r + 1, 2 * r + 1, 2).reshape(1, (2 * r + 1) ** 2, 2)
    for _ in range(B):
        with torch.no_grad():
            local_window_coords = (coords[_, :, :, None] + local_window[:, None, None]).reshape(1, h,
                                                                                                w * (2 * r + 1) ** 2, 2)
            window_feature = F.grid_sample(
                feature1[_:_ + 1], local_window_coords, padding_mode=padding_mode, align_corners=False,
                mode=sample_mode,  #
            )
            window_feature = window_feature.reshape(c, h, w, (2 * r + 1) ** 2)
        corr[_] = (feature0[_, ..., None] / (c ** .5) * window_feature).sum(dim=0).permute(2, 0, 1)
    return corr


def get_grid(b, h, w, device):
    grid = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device)
            for n in (b, h, w)
        ],
        indexing='ij'
    )
    grid = torch.stack((grid[2], grid[1]), dim=-1).reshape(b, h, w, 2)
    return grid


def get_autocast_params(device=None, enabled=False, dtype=None):
    if device is None:
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        # strip :X from device
        autocast_device = str(device).split(":")[0]
    if 'cuda' in str(device):
        out_dtype = dtype
        enabled = True
    else:
        out_dtype = torch.bfloat16
        enabled = False
        # mps is not supported
        autocast_device = "cpu"
    return autocast_device, enabled, out_dtype


def check_rgb(im):
    if im.mode != "RGB":
        raise NotImplementedError("Can't handle non-RGB images")


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
        if file_name.endswith(".obj"):
            obj_name = file_name
        elif file_name.endswith(".mtl"):
            material_name = file_name

    if obj_name is None or material_name is None:
        raise Exception(f"{data_dir} does not contain a valid obj file.")

    mtl_path = os.path.join(data_dir, material_name)
    texture_name = None
    with open(mtl_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('map_Kd'):
                # shlex handles quoted strings and spaces
                tokens = line.split(" ")
                if len(tokens) >= 2:
                    texture_name = tokens[-1]
    if texture_name is None:
        raise Exception(f"{data_dir} does not contain a valid obj file.")

    mesh_path = os.path.join(data_dir, obj_name)
    texture_path = os.path.join(data_dir, texture_name)
    mesh = trimesh.load(mesh_path, process=False)
    tex_img = Image.open(texture_path).resize((2048, 2048))  # 2k texture map for memory saving
    tex_img = np.asarray(tex_img).astype(np.float32) / 255.0

    verts = np.asarray(mesh.vertices).astype(np.float32)
    # verts[:, 2] += 0.8
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


def gen_orbit_views(azim_range=80, front_range=40):
    dist = 4.5
    elev = 0
    azims = np.linspace(-abs(azim_range), abs(azim_range), 40) / 180 * np.pi

    T_kgs = []
    T_gks = []
    view_mask = []
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
