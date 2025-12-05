import numpy as np
import cv2
import PIL
from PIL import Image
import os
from datetime import datetime
import pdb
import torch.nn.functional as F
import numpy as np
import os
import cv2
import copy
from scipy.interpolate import UnivariateSpline, interp1d
import numpy as np
import PIL.Image
import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional
import cv2
import PIL
import numpy
import skimage.io
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
import torchvision.transforms as T
import imageio
import re

def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor()
    ])
    tensor = transform(image)
    return tensor
    

def save_specific_frame(video_path, output_path, frame_id=0):
    """Save the specific frame of a video
    
    Args:
        video_path: Path to the input video
        output_path: Path to save the frame image
        frame_id: Frame index to extract (0-based, default is 0 for first frame)
    """
    cap = cv2.VideoCapture(video_path)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Saved frame {frame_id} to: {output_path}")
    else:
        raise ValueError(f"Failed to read frame {frame_id} from: {video_path}")


def parse_params(txt_path):
    """Parse relight parameters from txt file"""
    params = {
        'prompt': 'soft light',
        'bg_source': 'Left Light', 
        'cfg': 1.0,
        'steps': 25,
        'use_sky_mask': False,
        'input_bg_path': None
    }
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    parsed = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            parsed[key.strip()] = value.strip()

    if 'relight_prompt' in parsed:
        params['prompt'] = parsed['relight_prompt']
    
    if 'cfg' in parsed:
        try:
            params['cfg'] = float(parsed['cfg'].split()[0])
        except ValueError:
            pass

    if 'steps' in parsed:
        params['steps'] = parsed['steps']
    
    if 'use_sky_mask' in parsed:
        params['use_sky_mask'] = parsed['use_sky_mask']

    if 'input_bg_path' in parsed:
        params['input_bg_path'] = parsed['input_bg_path']
    
    if 'bg_source' in parsed:
        bg = parsed['bg_source'].split()[0]
        if is_likely_filepath(bg):
            params['bg_source'] = bg
        else:
            bg = bg.replace('_', ' ').title()
            params['bg_source'] = bg if bg.endswith('Light') else bg + ' Light'
                
    return params


def get_nearby_10_frames(input_video: torch.Tensor, ref_id: int) -> torch.Tensor:
    ref_id = int(ref_id)
    start_id = ref_id - 5
    end_id = ref_id + 5  # exclusive

    # Ensure indices are within valid range (video length fixed at 49)
    if start_id < 0:
        start_id = 0
        end_id = 10
    elif end_id > 49:
        end_id = 49
        start_id = 39

    return input_video[:, :, start_id:end_id]  # shape: [10, C, H, W]

def read_video_frames(video_path, process_length, stride, max_res, dataset="open"):
    if dataset == "open":
        print("==> processing video: ", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))

        # FIXME: hard coded
        width = 1024
        height = 576

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    if stride == -1:
        # Uniformly sample N frames from first to last frame
        if process_length <= 0:
            raise ValueError("When stride=-1, process_length must be positive")
        total_frames = len(vid)
        if process_length >= total_frames:
            frames_idx = list(range(total_frames))
        else:
            step = total_frames / process_length
            frames_idx = [int(i * step) for i in range(process_length)]
        print(f"==> uniformly sampling {len(frames_idx)} frames from {total_frames} total frames")
    else:
        # Original stride-based sampling
        frames_idx = list(range(0, len(vid), stride))
    
    print(
        f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with sampling method: {'uniform' if stride == -1 else f'stride {stride}'}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
    )
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

    return frames
    
def save_video(data, images_path, folder=None, fps=8):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder] * len(data)
        images = [
            np.array(Image.open(os.path.join(folder_name, path)))
            for folder_name, path in zip(folder, data)
        ]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(
        images_path, tensor_data, fps=fps, video_codec='h264', options={'crf': '10'}
    )


def sphere2pose(c2ws_input, theta, phi, r, device, x=None, y=None):
    c2ws = copy.deepcopy(c2ws_input)
    # c2ws[:,2, 3] = c2ws[:,2, 3] - radius

    # 先沿着世界坐标系z轴方向平移再旋转
    c2ws[:, 2, 3] -= r
    if x is not None:
        c2ws[:, 1, 3] += y
    if y is not None:
        c2ws[:, 0, 3] -= x

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = (
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, cos_value_x, -sin_value_x, 0],
                [0, sin_value_x, cos_value_x, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = (
        torch.tensor(
            [
                [cos_value_y, 0, sin_value_y, 0],
                [0, 1, 0, 0],
                [-sin_value_y, 0, cos_value_y, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    c2ws = torch.matmul(rot_mat_x, c2ws)
    c2ws = torch.matmul(rot_mat_y, c2ws)
    # c2ws[:,2, 3] = c2ws[:,2, 3] + radius
    return c2ws


def generate_traj_specified(c2ws_anchor, theta, phi, d_r, d_x, d_y, frame, device):
    # Initialize a camera.
    thetas = np.linspace(0, theta, frame)
    phis = np.linspace(0, phi, frame)
    rs = np.linspace(0, d_r, frame)
    xs = np.linspace(0, d_x, frame)
    ys = np.linspace(0, d_y, frame)
    c2ws_list = []
    for th, ph, r, x, y in zip(thetas, phis, rs, xs, ys):
        c2w_new = sphere2pose(
            c2ws_anchor,
            np.float32(th),
            np.float32(ph),
            np.float32(r),
            device,
            np.float32(x),
            np.float32(y),
        )
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)
    return c2ws


def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


def generate_traj_txt(c2ws_anchor, phi, theta, r, frame, device):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """

    if len(phi) > 3:
        phis = txt_interpolation(phi, frame, mode='smooth')
        phis[0] = phi[0]
        phis[-1] = phi[-1]
    else:
        phis = txt_interpolation(phi, frame, mode='linear')

    if len(theta) > 3:
        thetas = txt_interpolation(theta, frame, mode='smooth')
        thetas[0] = theta[0]
        thetas[-1] = theta[-1]
    else:
        thetas = txt_interpolation(theta, frame, mode='linear')

    if len(r) > 3:
        rs = txt_interpolation(r, frame, mode='smooth')
        rs[0] = r[0]
        rs[-1] = r[-1]
    else:
        rs = txt_interpolation(r, frame, mode='linear')
    # rs = rs*c2ws_anchor[0,2,3].cpu().numpy()

    c2ws_list = []
    for th, ph, r in zip(thetas, phis, rs):
        c2w_new = sphere2pose(
            c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r), device
        )
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)
    return c2ws


class Warper:
    def __init__(self, resolution: tuple = None, device: str = 'gpu0'):
        self.resolution = resolution
        self.device = self.get_device(device)
        self.dtype = torch.float32
        return

    def forward_warp(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor],
        mask=False,
        twice=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        All arrays should be torch tensors with batch dimension and channel first
        :param frame1: (b, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                        bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                        method accordingly.
        :param mask1: (b, 1, h, w) - 1 for known, 0 for unknown. Optional
        :param depth1: (b, 1, h, w)
        :param transformation1: (b, 4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (b, 4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (b, 3, 3) camera intrinsic matrix
        :param intrinsic2: (b, 3, 3) camera intrinsic matrix. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        assert frame1.shape == (b, 3, h, w)
        assert mask1.shape == (b, 1, h, w)
        assert depth1.shape == (b, 1, h, w)
        assert transformation1.shape == (b, 4, 4)
        assert transformation2.shape == (b, 4, 4)
        assert intrinsic1.shape == (b, 3, 3)
        assert intrinsic2.shape == (b, 3, 3)

        frame1 = frame1.to(self.device).to(self.dtype)
        mask1 = mask1.to(self.device).to(self.dtype)
        depth1 = depth1.to(self.device).to(self.dtype)
        transformation1 = transformation1.to(self.device).to(self.dtype)
        transformation2 = transformation2.to(self.device).to(self.dtype)
        intrinsic1 = intrinsic1.to(self.device).to(self.dtype)
        intrinsic2 = intrinsic2.to(self.device).to(self.dtype)

        trans_points1 = self.compute_transformed_points(
            depth1, transformation1, transformation2, intrinsic1, intrinsic2
        )
        trans_coordinates = (
            trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        )
        trans_depth1 = trans_points1[:, :, :, 2, 0]
        grid = self.create_grid(b, h, w).to(trans_coordinates)
        flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid
        if not twice:
            warped_frame2, mask2 = self.bilinear_splatting(
                frame1, mask1, trans_depth1, flow12, None, is_image=True
            )
            if mask:
                warped_frame2, mask2 = self.clean_points(warped_frame2, mask2)
            return warped_frame2, mask2, None, flow12

        else:
            warped_frame2, mask2 = self.bilinear_splatting(
                frame1, mask1, trans_depth1, flow12, None, is_image=True
            )
            # warped_frame2, mask2 = self.clean_points(warped_frame2, mask2)
            warped_flow, _ = self.bilinear_splatting(
                flow12, mask1, trans_depth1, flow12, None, is_image=False
            )
            twice_warped_frame1, _ = self.bilinear_splatting(
                warped_frame2,
                mask2,
                depth1.squeeze(1),
                -warped_flow,
                None,
                is_image=True,
            )
            return twice_warped_frame1, warped_frame2, None, None

    def compute_transformed_points(
        self,
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor],
    ):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        b, _, h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        transformation = torch.bmm(
            transformation2, torch.linalg.inv(transformation1)
        )  # (b, 4, 4)

        x1d = torch.arange(0, w)[None]
        y1d = torch.arange(0, h)[:, None]
        x2d = x1d.repeat([h, 1]).to(depth1)  # (h, w)
        y2d = y1d.repeat([1, w]).to(depth1)  # (h, w)
        ones_2d = torch.ones(size=(h, w)).to(depth1)  # (h, w)
        ones_4d = ones_2d[None, :, :, None, None].repeat(
            [b, 1, 1, 1, 1]
        )  # (b, h, w, 1, 1)
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[
            None, :, :, :, None
        ]  # (1, h, w, 3, 1)

        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
        trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

        unnormalized_pos = torch.matmul(
            intrinsic1_inv_4d, pos_vectors_homo
        )  # (b, h, w, 3, 1)
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
        trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
        return trans_norm_points

    def bilinear_splatting(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        flow12: torch.Tensor,
        flow12_mask: Optional[torch.Tensor],
        is_image: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame1)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack(
            [
                torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        trans_pos_floor = torch.stack(
            [
                torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        trans_pos_ceil = torch.stack(
            [
                torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = torch.moveaxis(
            prox_weight_nw * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_sw = torch.moveaxis(
            prox_weight_sw * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_ne = torch.moveaxis(
            prox_weight_ne * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_se = torch.moveaxis(
            prox_weight_se * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(
            frame1
        )
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(
            frame1
        )

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_nw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_sw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_ne,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_se,
            accumulate=True,
        )

        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            weight_nw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            weight_sw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            weight_ne,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            weight_se,
            accumulate=True,
        )

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(
            mask, cropped_warped_frame / cropped_weights, zero_tensor
        )
        mask2 = mask.to(frame1)

        if is_image:
            assert warped_frame2.min() >= -1.1  # Allow for rounding errors
            assert warped_frame2.max() <= 1.1
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        return warped_frame2, mask2

    def clean_points(self, warped_frame2, mask2):
        warped_frame2 = (warped_frame2 + 1.0) / 2.0
        mask = 1 - mask2
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0) * 255.0
        mask = mask.cpu().numpy()
        kernel = numpy.ones((5, 5), numpy.uint8)
        mask_erosion = cv2.dilate(numpy.array(mask), kernel, iterations=1)
        mask_erosion = PIL.Image.fromarray(numpy.uint8(mask_erosion))
        mask_erosion_ = numpy.array(mask_erosion) / 255.0
        mask_erosion_[mask_erosion_ < 0.5] = 0
        mask_erosion_[mask_erosion_ >= 0.5] = 1
        mask_new = (
            torch.from_numpy(mask_erosion_)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        warped_frame2 = warped_frame2 * (1 - mask_new)
        return warped_frame2 * 2.0 - 1.0, 1 - mask_new[:, 0:1, :, :]

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def read_image(path: Path) -> torch.Tensor:
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def read_depth(path: Path) -> torch.Tensor:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        else:
            raise RuntimeError(f'Unknown depth format: {path.suffix}')
        return depth

    @staticmethod
    def camera_intrinsic_transform(
        capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)
    ):
        start_y, start_x = patch_start_point
        camera_intrinsics = numpy.eye(4)
        camera_intrinsics[0, 0] = 2100
        camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
        camera_intrinsics[1, 1] = 2100
        camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
        return camera_intrinsics

    @staticmethod
    def get_device(device: str):
        """
        Returns torch device object
        :param device: cpu/gpu0/gpu1
        :return:
        """
        if device == 'cpu':
            device = torch.device('cpu')
        elif device.startswith('gpu') and torch.cuda.is_available():
            gpu_num = int(device[3:])
            device = torch.device(f'cuda:{gpu_num}')
        else:
            device = torch.device('cpu')
        return device

def is_likely_filepath(text):
    """
    Determine whether a string is likely a file path.
    
    Args:
        text (str): The string to check.
    
    Returns:
        bool: True if the string appears to be a file path, False otherwise.
    """
    # 1. Contains path separators
    if os.path.sep in text or ('\\' in text and os.name == 'nt'):
        return True
    
    # 2. Common file extensions
    file_extensions = [
        '.hdr', '.exr', '.jpg', '.jpeg', '.png', '.tif', '.tiff',
        '.tga', '.bmp', '.raw', '.dng', '.cr2', '.nef', '.obj',
        '.fbx', '.blend', '.ma', '.mb', '.gltf', '.glb'
    ]
    text_lower = text.lower()
    if any(text_lower.endswith(ext) for ext in file_extensions):
        return True
    
    # 3. Path pattern matching
    path_patterns = [
        r'^[A-Za-z]:[\\/]',  # Windows drive letter: C:\ or D:/
        r'^\.{1,2}[\\/]',    # Relative paths: ./ or ../
        r'^~[\\/]',          # User directory: ~/
        r'^[\\/]',           # Unix absolute path: /
        r'^[A-Za-z]:\\\\',   # Windows network path
    ]
    
    for pattern in path_patterns:
        if re.match(pattern, text):
            return True
    
    # 4. Check for common path components
    path_components = ['/', '\\', ':', '.hdr', '.exr', '.jpg', '.png']
    if any(comp in text for comp in path_components):
        # Further validation: try to extract possible extension
        if '.' in text:
            # Get the part after the last dot (likely extension)
            possible_ext = text.split('.')[-1].lower()
            common_exts = ['hdr', 'exr', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'tga', 'bmp']
            if possible_ext in common_exts:
                return True
    
    return False


def replace_background_with_image(foreground_video_path, mask_video_path, background_image_path, output_video_path, threshold=0.5):
    """
    Replace background regions (where mask < threshold) in foreground video with corresponding regions from background image
    
    Parameters:
    foreground_video_path: Path to foreground video
    mask_video_path: Path to mask video
    background_image_path: Path to background image
    output_video_path: Path to output video
    threshold: Threshold value, default is 0.5
    """
    
    # Open foreground video and mask video
    foreground_cap = cv2.VideoCapture(foreground_video_path)
    mask_cap = cv2.VideoCapture(mask_video_path)
    
    # Read background image
    background_image = cv2.imread(background_image_path)
    if background_image is None:
        print(f"Cannot open background image: {background_image_path}")
        return
    
    # Check if videos are successfully opened
    if not foreground_cap.isOpened():
        print(f"Cannot open foreground video: {foreground_video_path}")
        return
    
    if not mask_cap.isOpened():
        print(f"Cannot open mask video: {mask_video_path}")
        return
        
    # Get foreground video properties
    fps = int(foreground_cap.get(cv2.CAP_PROP_FPS))
    width = int(foreground_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(foreground_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(foreground_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get mask video properties
    mask_frames = int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mask_width = int(mask_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mask_height = int(mask_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Foreground video: {width}x{height}, {fps}fps, {total_frames} frames")
    print(f"Mask video: {mask_width}x{mask_height}, {mask_frames} frames")
    print(f"Background image: {background_image.shape[1]}x{background_image.shape[0]}")
    
    # Check if frame counts match
    min_frames = min(total_frames, mask_frames)
    if min_frames != total_frames:
        print(f"Warning: Video frame counts don't match exactly, will process using minimum frame count ({min_frames})")
    
    # Resize background image to match foreground video
    if background_image.shape[0] != height or background_image.shape[1] != width:
        background_image = cv2.resize(background_image, (width, height))
        print(f"Background image resized to: {width}x{height}")
    
    # Create video writer using imageio
    print("Starting video processing...")
    
    with imageio.get_writer(output_video_path, fps=fps, codec='libx264', quality=9) as writer:
        frame_count = 0
        
        # Create progress bar using tqdm
        pbar = tqdm(total=min_frames, desc="Processing progress")
        
        while frame_count < min_frames:
            # Read frames from foreground video and mask video
            ret_fg, foreground_frame = foreground_cap.read()
            ret_mask, mask_frame = mask_cap.read()
            
            if not ret_fg or not ret_mask:
                break
            
            # Resize mask video to match foreground video
            if mask_frame.shape[0] != height or mask_frame.shape[1] != width:
                mask_frame = cv2.resize(mask_frame, (width, height))
            
            # Process mask
            if len(mask_frame.shape) == 3:
                mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask_frame
            
            # Normalize mask to 0-1 range
            mask_normalized = mask_gray.astype(np.float32) / 255.0
            
            # Create background mask (regions where mask < threshold are 1, others are 0)
            background_mask = (mask_normalized < threshold).astype(np.uint8)
            
            # Create foreground mask (regions where mask >= threshold are 1, others are 0)
            foreground_mask = (mask_normalized >= threshold).astype(np.uint8)
            
            # Perform pixel-wise blending using masks
            result_frame = foreground_frame.copy()
            result_frame[background_mask > 0] = background_image[background_mask > 0]
            
            # Convert BGR to RGB (imageio uses RGB format)
            result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            
            # Write frame to output video
            writer.append_data(result_frame_rgb)
            
            frame_count += 1
            pbar.update(1)
            
            # Update progress information every 50 frames
            if frame_count % 50 == 0:
                pbar.set_postfix({"Processed": f"{frame_count}/{min_frames}"})
        
        pbar.close()
    
    # Release resources
    foreground_cap.release()
    mask_cap.release()
    
    print(f"Processing completed! Output video saved to: {output_video_path}")