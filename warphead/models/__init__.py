from typing import Union
from warphead.models.encoders import *
from warphead.models.decoders import *
from warphead.models.pipeline import Pipeline
from warphead.models.transformer import Block, MemEffAttention

weight_urls = {
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", #hopefully this doesnt change :D
}

def warphead_model(resolution, weights=None, **kwargs):
    # warphead weights and dinov2 weights are loaded seperately, as dinov2 weights are not parameters
    # torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul TODO: these probably ruin stuff, should be careful
    # torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = CoarseDecoder(
        nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]),
        decoder_dim,
        cls_to_coord_res ** 2 + 1,
        is_classifier=True,
        amp=True,
        pos_enc=False, )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True

    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius=7,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "8": ConvRefiner(
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius=3,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "4": ConvRefiner(
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius=2,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "2": ConvRefiner(
                2 * 64 + 16,
                128 + 16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict({
        "16": proj16,
        "8": proj8,
        "4": proj4,
        "2": proj2,
        "1": proj1,
    })
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Coarse2FineDecoder(coordinate_decoder,
                                 gps,
                                 proj,
                                 conv_refiner,
                                 detach=True,
                                 scales=["16", "8", "4", "2", "1"],
                                 displacement_dropout_p=displacement_dropout_p,
                                 gm_warp_dropout_p=gm_warp_dropout_p)

    encoder = Encoder(
        cnn_kwargs=dict(
            pretrained=False,
            amp=True),
        amp=True,
    )  # for image
    helper = Helper()

    h, w = resolution
    matcher = Pipeline(encoder, helper, decoder, h=h, w=w, **kwargs)
    matcher.load_state_dict(weights, strict=False)
    return matcher

def get_warphead(device, weights=None, resolution: Union[int,tuple[int,int]] = 448, amp_dtype: torch.dtype = torch.float32):
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    if str(device) == 'cpu':
        amp_dtype = torch.float32

    assert resolution[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
    assert resolution[1] % 14 == 0, "Needs to be multiple of 14 for backbone"

    dinov2_weights = torch.hub.load_state_dict_from_url(weight_urls["dinov2"], map_location=device)
    model = warphead_model(resolution=resolution, weights=weights, dinov2_weights=dinov2_weights, device=device, amp_dtype=amp_dtype)
    print(f"Using coarse resolution {resolution}")
    return model
