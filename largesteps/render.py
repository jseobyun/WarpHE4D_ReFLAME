import torch
import numpy as np
import nvdiffrast.torch as dr

class SphericalHarmonics:
    """
    Environment map approximation using spherical harmonics.

    This class implements the spherical harmonics lighting model of [Ramamoorthi
    and Hanrahan 2001], that approximates diffuse lighting by an environment map.
    """

    def __init__(self, envmap):
        """
        Precompute the coefficients given an envmap.

        Parameters
        ----------
        envmap : torch.Tensor
            The environment map to approximate.
        """
        h,w = envmap.shape[:2]

        # Compute the grid of theta, phi values
        theta = (torch.linspace(0, np.pi, h, device='cuda')).repeat(w, 1).t()
        phi = (torch.linspace(3*np.pi, np.pi, w, device='cuda')).repeat(h,1)

        # Compute the value of sin(theta) once
        sin_theta = torch.sin(theta)
        # Compute x,y,z
        # This differs from the original formulation as here the up axis is Y
        x = sin_theta * torch.cos(phi)
        z = -sin_theta * torch.sin(phi)
        y = torch.cos(theta)

        # Compute the polynomials
        Y_0 = 0.282095
        # The following are indexed so that using Y_n[-p]...Y_n[p] gives the proper polynomials
        Y_1 = [
            0.488603 * z,
            0.488603 * x,
            0.488603 * y
            ]
        Y_2 = [
            0.315392 * (3*z.square() - 1),
            1.092548 * x*z,
            0.546274 * (x.square() - y.square()),
            1.092548 * x*y,
            1.092548 * y*z
        ]
        import matplotlib.pyplot as plt
        area = w*h
        radiance = envmap[..., :3]
        dt_dp = 2.0 * np.pi**2 / area

        # Compute the L coefficients
        L = [ [(radiance * Y_0 * (sin_theta)[..., None] * dt_dp).sum(dim=(0,1))],
            [(radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0,1)) for y in Y_1],
            [(radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0,1)) for y in Y_2]]

        # Compute the R,G and B matrices
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743125
        c4 = 0.886227
        c5 = 0.247708

        self.M = torch.stack([
            torch.stack([ c1 * L[2][2] , c1 * L[2][-2], c1 * L[2][1] , c2 * L[1][1]           ]),
            torch.stack([ c1 * L[2][-2], -c1 * L[2][2], c1 * L[2][-1], c2 * L[1][-1]          ]),
            torch.stack([ c1 * L[2][1] , c1 * L[2][-1], c3 * L[2][0] , c2 * L[1][0]           ]),
            torch.stack([ c2 * L[1][1] , c2 * L[1][-1], c2 * L[1][0] , c4 * L[0][0] - c5 * L[2][0]])
        ]).movedim(2,0)

    def eval(self, n):
        """
        Evaluate the shading using the precomputed coefficients.

        Parameters
        ----------
        n : torch.Tensor
            Array of normals at which to evaluate lighting.
        """
        normal_array = n.view((-1, 3))
        h_n = torch.nn.functional.pad(normal_array, (0,1), 'constant', 1.0)
        l = (h_n.t() * (self.M @ h_n.t())).sum(dim=1)
        return l.t().view(n.shape)

def persp_proj(fov_x=45, ar=1, near=0.1, far=100):
    """
    Build a perspective projection matrix.

    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fov_x)
    proj_mat = np.array([[-1.0 / np.tan(fov_rad / 2.0), 0, 0, 0],
                      [0, np.float32(ar) / np.tan(fov_rad / 2.0), 0, 0],
                      [0, 0, -(near + far) / (near-far), 2 * far * near / (near-far)],
                      [0, 0, 1, 0]])
    x = torch.tensor([[1,2,3,4]], device='cuda')
    proj = torch.tensor(proj_mat, device='cuda', dtype=torch.float32)
    return proj

class NVDRenderer:
    """
    Renderer using nvdiffrast.


    This class encapsulates the nvdiffrast renderer [Laine et al 2020] to render
    objects given a number of viewpoints and rendering parameters.
    """
    def __init__(self, scene_params, shading=False, boost=1.0):
        """
        Initialize the renderer.

        Parameters
        ----------
        scene_params : dict
            The scene parameters. Contains the envmap and camera info.
        shading: bool
            Use shading in the renderings, otherwise render silhouettes. (default True)
        boost: float
            Factor by which to multiply shading-related gradients. (default 1.0)
        """
        # We assume all cameras have the same parameters (fov, clipping planes)
        near = scene_params["near_clip"]
        far = scene_params["far_clip"]
        self.fov_x = scene_params["fov"]
        w = scene_params["res_x"]
        h = scene_params["res_y"]
        self.res = (h,w)
        ar = w/h
        x = torch.tensor([[1,2,3,4]], device='cuda')
        self.proj_mat = persp_proj(self.fov_x, ar, near, far)

        # Construct the Model-View-Projection matrices
        self.view_mats = torch.stack(scene_params["view_mats"])
        self.mvps = self.proj_mat @ self.view_mats

        self.boost = boost
        self.shading = shading

        # Initialize rasterizing context
        self.glctx = dr.RasterizeGLContext()

        # Load the environment map
        w,h,_ = scene_params['envmap'].shape
        envmap = scene_params['envmap_scale'] * scene_params['envmap']
        # Precompute lighting
        self.sh = SphericalHarmonics(envmap)
        # Render background for all viewpoints once
        self.render_backgrounds(envmap)

    def render_backgrounds(self, envmap):
        """
        Precompute the background of each input viewpoint with the envmap.

        Params
        ------
        envmap : torch.Tensor
            The environment map used in the scene.
        """
        h,w = self.res
        pos_int = torch.arange(w*h, dtype = torch.int32, device='cuda')
        pos = 0.5 - torch.stack((pos_int % w, pos_int // w), dim=1) / torch.tensor((w,h), device='cuda')
        a = np.deg2rad(self.fov_x)/2
        r = w/h
        f = torch.tensor((2*np.tan(a),  2*np.tan(a)/r), device='cuda', dtype=torch.float32)
        rays = torch.cat((pos*f, torch.ones((w*h,1), device='cuda'), torch.zeros((w*h,1), device='cuda')), dim=1)
        rays_norm = (rays.transpose(0,1) / torch.norm(rays, dim=1)).transpose(0,1)
        rays_view = torch.matmul(rays_norm, self.view_mats.inverse().transpose(1,2)).reshape((self.view_mats.shape[0],h,w,-1))
        theta = torch.acos(rays_view[..., 1])
        phi = torch.atan2(rays_view[..., 0], rays_view[..., 2])
        envmap_uvs = torch.stack([0.75-phi/(2*np.pi), theta / np.pi], dim=-1)
        self.bgs = dr.texture(envmap[None, ...], envmap_uvs, filter_mode='linear').flip(1)
        self.bgs[..., -1] = 0 # Set alpha to 0
        # self.bgs *=0

    def project(self, v):
        v_c = v.clone()
        v_hom = torch.nn.functional.pad(v_c, (0, 1), 'constant', 1.0)  # [V, 4]
        v_ndc = torch.matmul(v_hom, self.mvps.transpose(1, 2))  # [B, V, 4]

        v_ndc = v_ndc[:, :, :3] / v_ndc[:, :, 3:4]
        x = (v_ndc[:, :, 0:1] * 0.5 + 0.5)
        y = (v_ndc[:, :, 1:2] * 0.5 + 0.5)
        xy_pixel = torch.cat([x, y], dim=2)  # (N, 2)
        return xy_pixel


    def render(self, v, n, f, tex_img=None, uv=None, uv_idx=None):
        """
        Render the scene in a differentiable way.

        Parameters
        ----------
        v : torch.Tensor
            Vertex positions, shape [V, 3]
        n : torch.Tensor
            Vertex normals, shape [V, 3]
        f : torch.Tensor
            Face indices, shape [F, 3]
        tex_img : torch.Tensor or None
            Texture image, shape [C, H_tex, W_tex], or None
        uv : torch.Tensor or None
            - If per-vertex UV: shape [V, 2]
            - If per-face   UV: shape [F, 3, 2]
            - Or None: no texture

        Returns
        -------
        result : torch.Tensor
            Rendered RGBA images, shape [B, H, W, 4]
        """
        # ───────────────────────────────────────────────────────────────
        # 1) 뷰 개수 B는 mvps의 첫 번째 차원에서 가져오기
        #    v/f는 단일 메쉬([V,3], [F,3])이므로 v.ndim == 2, f.ndim == 2

        B = int(self.mvps.shape[0])  # 예: 13


        # 2) f(face indices)를 torch.int32로 변환하여 같은 디바이스에 올리기
        f = f.to(v.device)  # 우선 v.device(CUDA)로 이동
        f = f.to(torch.int32)  # dtype을 int32로

        # 3) Homogeneous 좌표 변환 → NDC 투영
        #    v: [V, 3] → pad → [V, 4]
        v_hom = torch.nn.functional.pad(v, (0, 1), 'constant', 1.0)  # [V, 4]
        #    self.mvps: [B, 4, 4], v_hom: [V, 4] → matmul 브로드캐스트 결과: [B, V, 4]
        v_ndc = torch.matmul(v_hom, self.mvps.transpose(1, 2))  # [B, V, 4]

        # 4) rasterize → rast: [B, H, W, 4]
        #    dr.rasterize(...)도 (value, db) 튜플 반환 → 첫 번째 요소만 꺼냄

        rast, _ = dr.rasterize(self.glctx, v_ndc, f, self.res)  # rast: [B, H, W, 4]

        # 5) (옵션) SH 셰이딩 계산 → per-vertex SH → dr.interpolate → 픽셀별 조명
        if self.shading:
            # n: [V, 3] → 버텍스별 법선으로 SH eval → [V, 3] → (배치뷰가 없으므로) broadcast 불필요
            #    만약 SH 네트워크가 “배치 입력”을 기대한다면 n.unsqueeze(0).repeat(B,1,1) 형태로 넣으세요.
            vert_light = self.sh.eval(n).contiguous()  # [V, 3]
            #    broadcast: vert_light: [V,3] → [B, V, 3]로 간주됨
            #    dr.interpolate → (value, db) 튜플 반환
            light, _ = dr.interpolate(vert_light[None, ...], rast, f)  # light: [B, H, W, 3]
            # 위 코드에서 vert_light[None] → [1, V, 3] → 브로드캐스트 → [B, V, 3]로 처리됨
        else:
            light = None

        # 6) 텍스처 매핑
        if (tex_img is not None) and (uv is not None):
            # 6-A) 텍스처 이미지: [C, H_tex, W_tex] → 같은 디바이스로
            tex_img = tex_img.to(v.device).float()
            C, H_tex, W_tex = tex_img.shape

            # 6-B) uv 형태 분기
            vertex_uv = uv.to(v.device).float()  # [V, 2]
            # 배치 차원 추가 & 확장 → [B, V, 2]
            uv_batched = vertex_uv.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, V, 2]
            if uv_idx is None:
                pix_uv, _ = dr.interpolate(uv_batched, rast, f)  # pix_uv: [B, H, W, 2]
            elif uv_idx is not None:
                pix_uv, _ = dr.interpolate(uv_batched, rast, uv_idx)  # pix_uv: [B, H, W, 2]
            else:
                raise RuntimeError(
                    f"Invalid UV shape: {tuple(uv.shape)}. "
                    "Expected per-vertex UV ([V,2]) or per-face UV ([F,3,2])."
                )
            # 6-C) 텍스처 이미지도 배치 차원으로 확장 → [B, C, H_tex, W_tex]
            #      expand는 실제 메모리를 중복하지 않고 view 차원만 늘려주므로 안전
            tex_batched = tex_img.unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # [B, C, H_tex, W_tex]

            # 6-D) dr.texture 호출 → (tex_color, db) 튜플 반환
            tex_color = dr.texture(
                tex_batched,  # [B, C, H_tex, W_tex]
                pix_uv,  # [B, H, W, 2]
                filter_mode='linear',
                boundary_mode='clamp'
            )  # tex_color: [B, H, W, C]

            # 6-E) 텍스처 컬러 + 셰이딩 결합 (선택 사항)
            if self.shading and (light is not None):
                # light: [B, H, W, 3], tex_color: [B, H, W, C]
                # 일반적으로 C=3(RGB)이므로, 곱셈이 잘 broadcast됨
                col_rgb = tex_color * (light / np.pi)  # [B, H, W, 3]
            else:
                col_rgb = tex_color  # [B, H, W, C]

            # 6-F) 알파 채널 붙이기 → [B, H, W, 4]
            alpha = torch.ones((*col_rgb.shape[:-1], 1), device=col_rgb.device)  # [B, H, W, 1]
            col = torch.cat((col_rgb, alpha), dim=-1)  # [B, H, W, 4]

        else:
            # 텍스처/UV 정보가 없는 경우: 셰이딩만 (light / π), alpha=1
            if not self.shading:
                raise RuntimeError("tex_img 또는 uv가 모두 None인데, shading도 꺼진 상태입니다!")
            # light: [B, H, W, 3]
            col_rgb = light / np.pi
            alpha = torch.ones((*col_rgb.shape[:-1], 1), device=col_rgb.device)  # [B, H, W, 1]
            col = torch.cat((col_rgb, alpha), dim=-1)  # [B, H, W, 4]

        # ───────────────────────────────────────────────────────────────
        # 7) Antialias 단계 → 최종 출력
        #    첫 번째 인자: [B, H, W, 4] (col vs 배경색 비교)
        #    rast:         [B, H, W, 4]
        #    v_ndc:        [B, V, 4]
        #    f:            [F, 3]
        result = dr.antialias(
            torch.where(rast[..., -1:] != 0, col, self.bgs),  # [B, H, W, 4]
            rast,
            v_ndc,
            f,
            pos_gradient_boost=self.boost
        )
        # result.shape == [B, H, W, 4]
        mask = rast[..., -1:]

        v_ndc_z = v_ndc[..., 2:3].contiguous()  # [B, V, 1] → z-component only

        # rast[..., :3] == barycentric coordinates
        # rast[..., 3] == triangle index

        depth, _ = dr.interpolate(v_ndc_z, rast, f)  # [B, H, W, 1]

        # Optionally, mask out background (triangle idx == -1)
        valid_mask = (rast[..., 3:] >= 0)
        depth[~valid_mask] = 0  # or np.i
        return result, mask, depth
