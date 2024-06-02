import sys
sys.path.append('./')
from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j] == True:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

def process_images_as_batch(dicts, garm_imgs, garment_des_list, is_checked, is_checked_crop, denoise_steps, seed):
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    human_imgs = []
    masks = []
    human_imgs_orig = []
    pose_imgs = []

    for i in range(len(dicts)):
        garm_img = garm_imgs[i].convert("RGB").resize((768, 1024))
        human_img_orig = dicts[i].convert("RGB")

        if is_checked_crop:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768, 1024))
        else:
            human_img = human_img_orig.resize((768, 1024))

        if is_checked:
            keypoints = openpose_model(human_img.resize((384, 512)))
            model_parse, _ = parsing_model(human_img.resize((384, 512)))
            mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
            mask = mask.resize((768, 1024))
        else:
            mask = pil_to_binary_mask(dicts[i]['layers'][0].convert("RGB").resize((768, 1024)))

        mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

        args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))

        human_imgs.append(human_img)
        masks.append(mask)
        human_imgs_orig.append(human_img_orig)
        pose_imgs.append(pose_img)  # Adding pose_img to the list

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompts = [f"model is wearing "] * len(dicts)
                negative_prompts = ["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(dicts)
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompts,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompts,
                    )

                    prompt_embeds_c_list = []
                    #for garment_des in garment_des_list:
                    prompt = ["a photo of "] * len(garment_des_list)
                    negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(garment_des_list)
                    if not isinstance(prompt, list):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, list):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )
                        #prompt_embeds_c_list.append(prompt_embeds_c)

                    pose_imgs_tensors = torch.stack([tensor_transfrom(pose_img) for pose_img in pose_imgs]).to(device, torch.float16)
                    garm_tensors = torch.stack([tensor_transfrom(garm_img) for garm_img in garm_imgs]).to(device, torch.float16)
                    generators = [torch.Generator(device).manual_seed(seed) if seed is not None else None for _ in range(len(dicts))]
                    images = pipe(
                        prompt_embeds=prompt_embeds,  # Concatenating along the first dimension
                        negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),  # Concatenating along the first dimension
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),  # Concatenating along the first dimension
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),  # Concatenating along the first dimension
                        num_inference_steps=denoise_steps,
                        generator=generators,
                        strength=1.0,
                        pose_img=pose_imgs_tensors,  # Pass the pose_imgs tensor here
                        text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),  # Concatenating along the first dimension
                        cloth=garm_tensors,
                        mask_image=masks,
                        image=human_imgs,
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_imgs,
                        guidance_scale=2.0,
                    )[0]

    return images, masks

def start_tryon_batch(dicts, garm_imgs, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    results = process_images_as_batch(dicts, garm_imgs, garment_des, is_checked, is_checked_crop, denoise_steps, seed)
    return results


def start_tryon(dicts, garm_imgs, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    results = process_images_as_batch(dicts, garm_imgs, garment_des, is_checked, is_checked_crop, denoise_steps, seed)
    return results

garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path, "human"))
human_list_path = [os.path.join(example_path, "human", human) for human in human_list]
