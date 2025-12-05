import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from enum import Enum
from torch.hub import download_url_to_file
import cv2
import random
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from briarmbg import BriaRMBG
from da.depth_anything_v2.dpt import DepthAnythingV2

encoder = "vitl"
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
model_path = f'tools/iclight/da/checkpoints/depth_anything_v2_{encoder}.pth'
model_urls = {
    'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
    'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
    'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
}
# Download model if not exists
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if encoder in model_urls:
        url = model_urls[encoder]
        print(f"Downloading {encoder} model...")
        download_url_to_file(url=url, dst=model_path)
        print(f"Download completed: {model_path}")
    else:
        raise ValueError(f"Unsupported encoder type: {encoder}")
# Load model
depth_anything = DepthAnythingV2(**model_configs[encoder])
depth_anything.load_state_dict(torch.load(model_path, map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()


sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Change UNet
with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward

def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

unet.forward = hooked_unet_forward

# Load

model_path = './tools/iclight/checkpoints/iclight_sd15_fc.safetensors'
os.makedirs(os.path.dirname(model_path), exist_ok=True)

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

# t2i_pipe = StableDiffusionPipeline(
#     vae=vae,
#     text_encoder=text_encoder,
#     tokenizer=tokenizer,
#     unet=unet,
#     scheduler=dpmpp_2m_sde_karras_scheduler,
#     safety_checker=None,
#     requires_safety_checker=False,
#     feature_extractor=None,
#     image_encoder=None
# )

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, sky_mask, input_bg_path=None):
    bg_source = BGSource(bg_source)
    input_bg = None

    if bg_source == BGSource.NONE:
        input_bg = None
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width, dtype=np.uint8)
        input_bg = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((input_bg,) * 3, axis=-1)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width, dtype=np.uint8)
        input_bg = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((input_bg,) * 3, axis=-1)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height, dtype=np.uint8)[:, None]
        input_bg = np.tile(gradient, (1, image_width))
        input_bg = np.stack((input_bg,) * 3, axis=-1)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height, dtype=np.uint8)[:, None]
        input_bg = np.tile(gradient, (1, image_width))
        input_bg = np.stack((input_bg,) * 3, axis=-1)
    elif bg_source in [BGSource.TOP_LEFT, BGSource.TOP_RIGHT, BGSource.BOTTOM_LEFT, BGSource.BOTTOM_RIGHT]:
        # Horizontal direction
        if bg_source in [BGSource.TOP_LEFT, BGSource.BOTTOM_LEFT]:
            grad_x = np.linspace(255, 0, image_width, dtype=np.float32)
        else:
            grad_x = np.linspace(0, 255, image_width, dtype=np.float32)
        
        # Vertical direction
        if bg_source in [BGSource.TOP_LEFT, BGSource.TOP_RIGHT]:
            grad_y = np.linspace(255, 0, image_height, dtype=np.float32)[:, None]
        else:
            grad_y = np.linspace(0, 255, image_height, dtype=np.float32)[:, None]
        
        grad_x = np.tile(grad_x, (image_height, 1))
        grad_y = np.tile(grad_y, (1, image_width))
        
        gradient = ((grad_x + grad_y) / 2).astype(np.uint8)  # Simple average blending
        input_bg = np.stack((gradient,) * 3, axis=-1)
    
    elif bg_source == BGSource.RANDOM:
        # Generate random 2D linear gradient
        angle = random.uniform(0, 2 * np.pi)
        x = np.linspace(-1, 1, image_width)
        y = np.linspace(-1, 1, image_height)
        xx, yy = np.meshgrid(x, y)
        gradient = np.cos(angle) * xx + np.sin(angle) * yy
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min()) * 255
        gradient = gradient.astype(np.uint8)
        input_bg = np.stack((gradient,) * 3, axis=-1)
    elif bg_source == BGSource.Custom:
        if input_bg_path is None:
            raise ValueError("BGSource.Custom mode requires `input_bg_path` to be provided.")
        
        # Read image from path
        input_bg = cv2.imread(input_bg_path, cv2.IMREAD_UNCHANGED)
        if input_bg is None:
            raise FileNotFoundError(f"Failed to read background image from {input_bg_path}")
        
        # Resize to target dimensions
        input_bg = cv2.resize(input_bg, (image_width, image_height))
        
        # Format normalization: grayscale / single channel → RGB
        if input_bg.ndim == 2:
            input_bg = np.stack((input_bg,) * 3, axis=-1)
        elif input_bg.shape[2] == 1:
            input_bg = np.concatenate([input_bg] * 3, axis=-1)
        elif input_bg.shape[2] == 4:
            input_bg = input_bg[:, :, :3]  # Discard alpha channel
        elif input_bg.shape[2] != 3:
            raise ValueError("Provided image must have 1, 3, or 4 channels.")
        input_bg = input_bg.astype(np.uint8)
    else:
        raise ValueError('Invalid bg_source provided.')
    
    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)
    
    if input_bg is None:
        latents = t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    
    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    if sky_mask is not None:
        sky_mask = (sky_mask > 128).astype(np.float32)
        sky_mask = sky_mask[..., None]
        pixels[0] = (1 - sky_mask) * pixels[0] + sky_mask * fg
        pixels[0] = pixels[0].astype(np.uint8)
    
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]
    
    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample

    return pytorch2numpy(pixels), input_bg


@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, sky_mask, input_bg_path=None):
    # input_fg, matting = run_rmbg(input_fg)
    results, input_bg = process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source, sky_mask, input_bg_path)
    return input_fg, results, input_bg


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    TOP_LEFT = "Top Left Light"
    TOP_RIGHT = "Top Right Light"
    BOTTOM_LEFT = "Bottom Left Light"
    BOTTOM_RIGHT = "Bottom Right Light"
    RANDOM = "Random Light"
    Custom = "Custom Light"

def relight_image(
    input_image,
    prompt="soft light",
    bg_source="Left Light",
    cfg=1.0,
    steps=25,
    highres_scale=1.5,
    highres_denoise=0.5,
    lowres_denoise=0.9,
    num_samples=1,
    seed=12345,
    a_prompt="best quality",
    n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
    use_sky_mask=False,
    input_bg_path=None
):
    """
    Relighting function for single image
    
    Args:
        input_image (numpy.ndarray): Input image as numpy array
        prompt (str): Lighting prompt for relighting
        bg_source (str): Type of background lighting source
        cfg (float): Classifier-free guidance scale
        steps (int): Number of sampling steps
        highres_scale (float): High-resolution upscaling factor
        highres_denoise (float): High-resolution denoise strength
        lowres_denoise (float): Low-resolution denoise strength
        num_samples (int): Number of generated samples per image
        seed (int): Random seed for reproducibility
        a_prompt (str): Positive prompt
        n_prompt (str): Negative prompt
        input_bg_path (str): Path to input background image (only used when bg_source is "Input Image")
        
    Returns:
        numpy.ndarray: Relit image as numpy array
    """
    
    # Read input image from numpy array
    input_fg = input_image
    image_height, image_width = input_fg.shape[:-1]

    # Infer depth and generate sky mask
    sky_mask = None
    if use_sky_mask:
        depth = depth_anything.infer_image(input_fg).astype(np.float32)
        sky_mask = (depth == 0).astype(np.uint8) * 255

    # Perform relighting
    input_fg, results, input_bg = process_relight(
        input_fg,
        prompt,
        image_width,
        image_height,
        num_samples,
        seed,
        steps,
        a_prompt,
        n_prompt,
        cfg,
        highres_scale,
        highres_denoise,
        lowres_denoise,
        bg_source,
        sky_mask,
        input_bg_path
    )

    relit_img = results[0]
    relit_img_resized = cv2.resize(relit_img, (input_fg.shape[1], input_fg.shape[0]), interpolation=cv2.INTER_AREA)
    relit_img_resized = np.clip(relit_img_resized, 0, 255).astype(np.uint8)

    print("Image relighting completed")
    
    return relit_img_resized


def main():
    parser = argparse.ArgumentParser(description="Batch relighting script for images")

    # Path and basic parameters
    parser.add_argument("--data_root", type=str,
                        default="",
                        help="Input folder containing .png images")
    parser.add_argument("--out_dir", type=str, default="",
                        help="Directory to save relit results")
    parser.add_argument("--input_bg_path", type=str, default=None,
                        help="Path to input background image (only used when --bg_source is 'Input Image')")

    # Lighting and background settings
    parser.add_argument("--prompt", type=str, default="sunlight",
                        help="Lighting prompt for relighting")
    # 'soft light', 'neon light', 'golden time', 'sci-fi RGB glowing, cyberpunk', 'natural lighting', 'magic lit',
    # 'evil, gothic, Yharnam', 'light and shadow', 'soft studio lighting', 'home atmosphere, cozy bedroom illumination', 'neon, Wong Kar-wai, warm'
    parser.add_argument("--bg_source", type=str, default="Right Light",
                        choices=[
                            "Left Light", "Right Light", "Top Light",
                            "Bottom Light", "Top Left Light", "Top Right Light",
                            "Bottom Left Light", "Bottom Right Light",
                            "Random Light", "Custom Light"
                        ],
                        help="Type of background lighting source")

    # Model inference parameters
    parser.add_argument("--cfg", type=float, default=2.0,#1.0
                        help="Classifier-free guidance scale")
    parser.add_argument("--steps", type=int, default=25,#10
                        help="Number of sampling steps")
    parser.add_argument("--highres_scale", type=float, default=1.5,
                        help="High-resolution upscaling factor")
    parser.add_argument("--highres_denoise", type=float, default=0.5,
                        help="High-resolution denoise strength")
    parser.add_argument("--lowres_denoise", type=float, default=0.9,
                        help="Low-resolution denoise strength")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of generated samples per image")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed for reproducibility")
    parser.add_argument("--use_sky_mask", action="store_true",
                        help="Whether to use sky mask based on depth estimation")

    args = parser.parse_args()

    # Fixed prompts
    a_prompt = "best quality"
    n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

    os.makedirs(args.out_dir, exist_ok=True)

    # Process all image files
    for scene_file in os.listdir(args.data_root):
        if not scene_file.endswith(".png"):
            continue

        scene = scene_file.split('.')[0]
        input_fg_path = os.path.join(args.data_root, scene_file)
        save_path = os.path.join(args.out_dir, f"relight_{scene}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Read input image
        input_fg = np.array(Image.open(input_fg_path).convert("RGB"))
        image_height, image_width = input_fg.shape[:-1]

        # Infer depth and generate sky mask
        sky_mask = None
        if args.use_sky_mask:
            depth = depth_anything.infer_image(input_fg).astype(np.float32)
            sky_mask = (depth == 0).astype(np.uint8) * 255

        # Perform relighting
        input_fg, results, input_bg = process_relight(
            input_fg,
            args.prompt,
            image_width,
            image_height,
            args.num_samples,
            args.seed,
            args.steps,
            a_prompt,
            n_prompt,
            args.cfg,
            args.highres_scale,
            args.highres_denoise,
            args.lowres_denoise,
            args.bg_source,
            sky_mask,
            args.input_bg_path
        )

        relit_img = results[0]
        relit_img_resized = cv2.resize(relit_img, (input_fg.shape[1], input_fg.shape[0]), interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_path, relit_img_resized[..., ::-1])
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
