# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import tempfile
import numpy as np
from . import util
from .body import Body, BodyResult, Keypoint
from .hand import Hand
from .face import Face
from .types import PoseResult, HandResult, FaceResult, AnimalPoseResult
from huggingface_hub import constants, hf_hub_download
from contextlib import suppress
from pathlib import Path
from .wholebody import Wholebody
import warnings
import cv2
from PIL import Image
from ast import literal_eval

from typing import Tuple, List, Union, Optional

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

UPSCALE_METHODS = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]
def get_upscale_method(method_str):
    assert method_str in UPSCALE_METHODS, f"Method {method_str} not found in {UPSCALE_METHODS}"
    return getattr(cv2, method_str)

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def resize_image_with_pad(input_image, resolution, upscale_method = "", skip_hwc3=False, mode='edge'):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad


def common_input_validate(input_image, output_type, **kwargs):
    if "img" in kwargs:
        warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
        input_image = kwargs.pop("img")

    if "return_pil" in kwargs:
        warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
        output_type = "pil" if kwargs["return_pil"] else "np"

    if type(output_type) is bool:
        warnings.warn(
            "Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
        if output_type:
            output_type = "pil"

    if input_image is None:
        raise ValueError("input_image must be defined.")

    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
        output_type = output_type or "pil"
    else:
        output_type = output_type or "np"

    return (input_image, output_type)


temp_dir = tempfile.gettempdir()
annotator_ckpts_path = os.path.join(Path(__file__).parents[2], 'ckpts')
USE_SYMLINKS = False

try:
    annotator_ckpts_path = os.environ['AUX_ANNOTATOR_CKPTS_PATH']
except:
    warnings.warn("Custom pressesor model path not set successfully.")
    pass

try:
    USE_SYMLINKS = literal_eval(os.environ['AUX_USE_SYMLINKS'])
except:
    warnings.warn("USE_SYMLINKS not set successfully. Using default value: False to download models.")
    pass

try:
    temp_dir = os.environ['AUX_TEMP_DIR']
    if len(temp_dir) >= 60:
        warnings.warn(f"custom temp dir is too long. Using default")
        temp_dir = tempfile.gettempdir()
except:
    warnings.warn(f"custom temp dir not set successfully")
    pass

here = Path(__file__).parent.resolve()

def custom_hf_download(pretrained_model_or_path, filename, cache_dir=temp_dir, ckpts_dir=annotator_ckpts_path, subfolder='', use_symlinks=USE_SYMLINKS, repo_type="model"):

    local_dir = os.path.join(ckpts_dir, pretrained_model_or_path)
    model_path = Path(local_dir).joinpath(*subfolder.split('/'), filename).__str__()

    if len(str(model_path)) >= 255:
        warnings.warn(f"Path {model_path} is too long, \n please change annotator_ckpts_path in config.yaml")

    if not os.path.exists(model_path):
        print(f"Failed to find {model_path}.\n Downloading from huggingface.co")
        print(f"cacher folder is {cache_dir}, you can change it by custom_tmp_path in config.yaml")
        if use_symlinks:
            cache_dir_d = constants.HF_HUB_CACHE    # use huggingface newer env variables `HF_HUB_CACHE`
            if cache_dir_d is None:
                import platform
                if platform.system() == "Windows":
                    cache_dir_d = Path(os.getenv("USERPROFILE")).joinpath(".cache", "huggingface", "hub").__str__()
                else:
                    cache_dir_d = os.path.join(os.getenv("HOME"), ".cache", "huggingface", "hub")
            try:
                # test_link
                Path(cache_dir_d).mkdir(parents=True, exist_ok=True)
                Path(ckpts_dir).mkdir(parents=True, exist_ok=True)
                (Path(cache_dir_d) / f"linktest_{filename}.txt").touch()
                # symlink instead of link avoid `invalid cross-device link` error.
                os.symlink(os.path.join(cache_dir_d, f"linktest_{filename}.txt"), os.path.join(ckpts_dir, f"linktest_{filename}.txt"))
                print("Using symlinks to download models. \n",\
                      "Make sure you have enough space on your cache folder. \n",\
                      "And do not purge the cache folder after downloading.\n",\
                      "Otherwise, you will have to re-download the models every time you run the script.\n",\
                      "You can use USE_SYMLINKS: False in config.yaml to avoid this behavior.")
            except:
                print("Maybe not able to create symlink. Disable using symlinks.")
                use_symlinks = False
                cache_dir_d = Path(cache_dir).joinpath("ckpts", pretrained_model_or_path).__str__()
            finally:    # always remove test link files
                with suppress(FileNotFoundError):
                    os.remove(os.path.join(ckpts_dir, f"linktest_{filename}.txt"))
                    os.remove(os.path.join(cache_dir_d, f"linktest_{filename}.txt"))
        else:
            cache_dir_d = os.path.join(cache_dir, "ckpts", pretrained_model_or_path)

        model_path = hf_hub_download(repo_id=pretrained_model_or_path,
            cache_dir=cache_dir_d,
            local_dir=local_dir,
            subfolder=subfolder,
            filename=filename,
            local_dir_use_symlinks=use_symlinks,
            resume_download=True,
            etag_timeout=100,
            repo_type=repo_type
        )
        if not use_symlinks:
            try:
                import shutil
                shutil.rmtree(os.path.join(cache_dir, "ckpts"))
            except Exception as e :
                print(e)

    print(f"model_path is {model_path}")

    return model_path


def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True, xinsr_stick_scaling=False):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints, xinsr_stick_scaling)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas


def decode_json_as_poses(
    pose_json: dict,
) -> Tuple[List[PoseResult], List[AnimalPoseResult], int, int]:
    """Decode the json_string complying with the openpose JSON output format
    to poses that controlnet recognizes.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md

    Args:
        json_string: The json string to decode.

    Returns:
        human_poses
        animal_poses
        canvas_height
        canvas_width
    """
    height = pose_json["canvas_height"]
    width = pose_json["canvas_width"]

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def decompress_keypoints(
        numbers: Optional[List[float]],
    ) -> Optional[List[Optional[Keypoint]]]:
        if not numbers:
            return None

        assert len(numbers) % 3 == 0

        def create_keypoint(x, y, c):
            if c < 1.0:
                return None
            keypoint = Keypoint(x, y)
            return keypoint

        return [create_keypoint(x, y, c) for x, y, c in chunks(numbers, n=3)]

    return (
        [
            PoseResult(
                body=BodyResult(
                    keypoints=decompress_keypoints(pose.get("pose_keypoints_2d"))
                ),
                left_hand=decompress_keypoints(pose.get("hand_left_keypoints_2d")),
                right_hand=decompress_keypoints(pose.get("hand_right_keypoints_2d")),
                face=decompress_keypoints(pose.get("face_keypoints_2d")),
            )
            for pose in pose_json.get("people", [])
        ],
        [decompress_keypoints(pose) for pose in pose_json.get("animals", [])],
        height,
        width,
    )


def encode_poses_as_dict(poses: List[PoseResult], canvas_height: int, canvas_width: int) -> str:
    """ Encode the pose as a dict following openpose JSON output format:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    """
    def compress_keypoints(keypoints: Union[List[Keypoint], None]) -> Union[List[float], None]:
        if not keypoints:
            return None
        
        return [
            value
            for keypoint in keypoints
            for value in (
                [float(keypoint.x), float(keypoint.y), float(keypoint.score)]
                if keypoint is not None
                else [0.0, 0.0, 0.0]
            )
        ]

    return {
        'people': [
            {
                'pose_keypoints_2d': compress_keypoints(pose.body.keypoints),
                "face_keypoints_2d": compress_keypoints(pose.face),
                "hand_left_keypoints_2d": compress_keypoints(pose.left_hand),
                "hand_right_keypoints_2d":compress_keypoints(pose.right_hand),
            }
            for pose in poses
        ],
        'canvas_height': canvas_height,
        'canvas_width': canvas_width,
    }

global_cached_dwpose = Wholebody()

class DwposeDetector:
    """
    A class for detecting human poses in images using the Dwpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """
    def __init__(self, dw_pose_estimation):
        self.dw_pose_estimation = dw_pose_estimation
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, pretrained_det_model_or_path=None, det_filename=None, pose_filename=None, torchscript_device="cuda"):
        global global_cached_dwpose
        pretrained_det_model_or_path = pretrained_det_model_or_path or pretrained_model_or_path
        det_filename = det_filename or "yolox_l.onnx"
        pose_filename = pose_filename or "dw-ll_ucoco_384.onnx"
        det_model_path = custom_hf_download(pretrained_det_model_or_path, det_filename)
        pose_model_path = custom_hf_download(pretrained_model_or_path, pose_filename)
        
        print(f"\nDWPose: Using {det_filename} for bbox detection and {pose_filename} for pose estimation")
        if global_cached_dwpose.det is None or global_cached_dwpose.det_filename != det_filename:
            t = Wholebody(det_model_path, None, torchscript_device=torchscript_device)
            t.pose = global_cached_dwpose.pose
            t.pose_filename = global_cached_dwpose.pose
            global_cached_dwpose = t
        
        if global_cached_dwpose.pose is None or global_cached_dwpose.pose_filename != pose_filename:
            t = Wholebody(None, pose_model_path, torchscript_device=torchscript_device)
            t.det = global_cached_dwpose.det
            t.det_filename = global_cached_dwpose.det_filename
            global_cached_dwpose = t
        return cls(global_cached_dwpose)

    def detect_poses(self, oriImg) -> List[PoseResult]:
        with torch.no_grad():
            keypoints_info = self.dw_pose_estimation(oriImg.copy())
            return Wholebody.format_result(keypoints_info)
    
    def __call__(self, input_image, detect_resolution=512, include_body=True, include_hand=False, include_face=False, hand_and_face=None, output_type="pil", image_and_json=False, upscale_method="INTER_CUBIC", xinsr_stick_scaling=False, **kwargs):
        if hand_and_face is not None:
            warnings.warn("hand_and_face is deprecated. Use include_hand and include_face instead.", DeprecationWarning)
            include_hand = hand_and_face
            include_face = hand_and_face

        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, _ = resize_image_with_pad(input_image, 0, upscale_method)
        poses = self.detect_poses(input_image)
        
        canvas = draw_poses(poses, input_image.shape[0], input_image.shape[1], draw_body=include_body, draw_hand=include_hand, draw_face=include_face, xinsr_stick_scaling=xinsr_stick_scaling)
        canvas, remove_pad = resize_image_with_pad(canvas, detect_resolution, upscale_method)
        detected_map = HWC3(remove_pad(canvas))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        
        if image_and_json:
            return (detected_map, encode_poses_as_dict(poses, input_image.shape[0], input_image.shape[1]))
        
        return detected_map
