import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from warphead.utils import check_rgb

class Pipeline(nn.Module):
    def __init__(
            self,
            encoder,
            helper,
            decoder,
            h=448,
            w=448,
            sample_mode="threshold_balanced",
            name=None,
            attenuate_cert=None,
            **kwargs,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.helper = helper
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h

        self.sample_mode = sample_mode
        self.sample_thresh = 0.05
        self.transforms = self.get_transform_ops(resize=(self.h_resized, self.w_resized), normalize=True)

    def get_transform_ops(self, resize=None, normalize=True):
        ops = []

        if resize:
            ops.append(transforms.Resize(resize))  # resize: (H, W)

        ops.append(transforms.ToTensor())  # Converts to [0,1] tensor

        if normalize:
            ops.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))  # Imagenet stats

        return transforms.Compose(ops)

    def get_output_resolution(self):
        return self.h_resized, self.w_resized

    def extract_backbone_features(self, batch):
        x_q = batch["im_A"]
        feature_pyramid = self.encoder(x_q), self.helper()
        return feature_pyramid

    def forward(self, batch, scale_factor=1):
        feature_pyramid = self.extract_backbone_features(batch)
        f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(f_q_pyramid,
                                f_s_pyramid,
                                scale_factor=scale_factor)
        return corresps

    @torch.inference_mode()
    def inference(
            self,
            im_A_input,
            device=None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        check_rgb(im_A_input)
        im_A = im_A_input

        self.train(False)
        with torch.no_grad():
            b = 1
            ws = self.w_resized
            hs = self.h_resized

            im_A = self.transforms(im_A)
            batch = {"im_A": im_A[None].to(device)}

            finest_scale = 1
            corresps = self.forward(batch)

            hs, ws = self.get_output_resolution()

            if self.attenuate_cert:
                low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            img_to_uv = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
            if finest_scale != 1:
                img_to_uv = F.interpolate(
                    img_to_uv, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = F.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            img_to_uv = img_to_uv.permute(
                0, 2, 3, 1
            )
            certainty = certainty.sigmoid()  # logits -> probs
            if (img_to_uv.abs() > 1).any() and True:
                wrong = (img_to_uv.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            img_to_uv = torch.clamp(img_to_uv, -1, 1)

            warp = img_to_uv
            return (
                warp[0],
                certainty[0, 0],
            )
