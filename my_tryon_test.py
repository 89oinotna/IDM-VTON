# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from ip_adapter.ip_adapter import Resampler

import argparse
import logging
import os
import torch.utils.data as data
import torchvision
import json
import accelerate
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline,DPMSolverMultistepScheduler
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer

from diffusers.utils.import_utils import is_xformers_available

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline



class TryOn():
    def __init__(self):
        self.clip_processor = CLIPImageProcessor()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.toTensor = transforms.ToTensor()
        self.model_path = "/group_share/model/IDM-VTON/"
        #/root/kj_work/IDM-VTON/local_directory/models--yisol--IDM-VTON/snapshots/585a32e74aee241cbc0d0cc3ab21392ca58c916a/

        self.vae = AutoencoderKL.from_pretrained(
            self.model_path,
            subfolder = "vae",
            torch_dtype = torch.float16,
            # use_safetensors=True
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_path,
            subfolder = "unet",
            torch_dtype = torch.float16
        )
        self.UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            self.model_path,
            subfolder = "unet_encoder",
            torch_dtype = torch.float16
        )#.to('cuda')
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            self.model_path,
            subfolder = "text_encoder",
            torch_dtype = torch.float16
        )
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            self.model_path,
            subfolder = "tokenizer",
            revision = None,
            use_fast = False
        )

        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.model_path,
            subfolder = "text_encoder_2",
            torch_dtype = torch.float16
        )
        self.tokenizer_two = CLIPTokenizer.from_pretrained(
            self.model_path,
            subfolder = "tokenizer_2",
            revision = None,
            use_fast = False
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.model_path,
            subfolder = "image_encoder",
            torch_dtype = torch.float16
        )
       
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_path,
            subfolder = "scheduler"
        )

        # print('1'*100)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.UNet_Encoder.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.UNet_Encoder.to('cuda', torch.float16)
        self.unet.eval()
        self.UNet_Encoder.eval()
        self.pipe = TryonPipeline.from_pretrained(
            self.model_path,
            unet = self.unet,
            vae = self.vae,
            feature_extractor = CLIPImageProcessor(),
            text_encoder = self.text_encoder_one,
            tokenizer = self.tokenizer_one,
            text_encoder_2 = self.text_encoder_two,
            tokenizer_2 = self.tokenizer_two,
            scheduler = self.noise_scheduler,
            image_encoder=self.image_encoder,
            torch_dtype = torch.float16,
        ).to('cuda')
        self.pipe.unet_encoder = self.UNet_Encoder
        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        #     self.pipe.scheduler.config,
        #     algorithm_type="sde-dpmsolver++",
        #     # use_karras_sigmas=True,
        # )
        # print('2'*100)

    def tryon(self, p1, p2, pose_img, cloth_img, img, mask):
        negative_prompt = "monochrome, lowre, bad anatomy, worsr quality, low quality, blurred, low resolution"
        
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        ) = self.pipe.encode_prompt(
            p1,
            num_images_per_prompt = 1, 
            do_classifier_free_guidance = True,
            negative_prompt = negative_prompt,
        )

        (
            prompt_embeds_c, _,_,_,
        ) = self.pipe.encode_prompt(
            p2, 
            num_images_per_prompt = 1,
            do_classifier_free_guidance = False,
            negative_prompt = negative_prompt
        )

        _cloth = self.clip_processor(images=cloth_img, return_tensors="pt").pixel_values
        img_emb_list = []
        for i in range(_cloth.shape[0]):
            img_emb_list.append(_cloth[i])
        image_embeds = torch.cat(img_emb_list,dim=0)
        # pose=self.transform(pose_img).shape
        # print(image_embeds.shape)
        # exit()
        generator = torch.Generator(self.pipe.device).manual_seed(42)
        # self.pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        # self.pipe.set_ip_adapter_scale(0.4)
        image = self.pipe(
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            pooled_prompt_embeds = pooled_prompt_embeds,
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
            num_inference_steps = 30,
            strength = 1,
            pose_img = torch.unsqueeze(self.transform(pose_img),0),
            text_embeds_cloth = prompt_embeds_c,
            cloth = torch.unsqueeze(self.transform(cloth_img),0).to(torch.float16).to('cuda'),
            mask_image = self.toTensor(mask)[:1],
            image = (self.transform(img)+1.0)/2.0,
            height = 1024,
            width = 768,
            guidance_scale = 2.0,
            ip_adapter_image = torch.unsqueeze(image_embeds,0),
            generator=generator
            # clip_skip=2,
        )
        # print(type(image[0][0]))
        return image[0][0]

if __name__ == "__main__":
    img_o = Image.open('my_pre_data/img/img1.jpg')

    from preprocess.openpose.run_openpose import OpenPose
    model = OpenPose(0,'/group_share/model/IDM-VTON/openpose/ckpts/')
    # keypoints=model('/root/kj_work/IDM-VTON/my_pre_data/img/img1.jpg')
    keypoints=model(img_o.copy())
    # print(keypoints)

    from preprocess.humanparsing.run_parsing import Parsing
    p = Parsing(0,'/group_share/model/IDM-VTON/humanparsing')
    # img, mask,parsed = p('/root/kj_work/IDM-VTON/my_pre_data/img')
    parsed = p(img_o.copy())
    print(parsed.shape)

    from my_get_maks import get_img_agnostic3
    # img = Image.open('my_pre_data/img/img1.jpg')
    pose_data = np.array(keypoints['pose_keypoints_2d'])
    # pose_data = pose_data.reshape((1, -1))[0]
    # print(pose_data)
    # exit()
    # pose_data = pose_data.reshape((-1, 2))
    # print(pose_data)
    # exit()
    agnostic = get_img_agnostic3(img_o.copy(), parsed, pose_data)
    agnostic.save('/root/kj_work/idm_output/mask.jpg')
    # exit()

    # print('1'*100)
    TO = TryOn()
    # print(2)
    p1 = ["Masterpiece,best quality,model is wearing a Over the knee down jacket"]# 衣服的种类，由LLM或者数据库给出
    p2 = ["Masterpiece,best quality,a photo of Over the knee down jacket"]# 衣服的种类，由LLM或者数据库给出
    
    # img = Image.open('my_pre_data/img/img1.jpg')
    cloth = Image.open('/root/kj_work/IDM-VTON/my_pre_data/cloth/c3.jpg')
    # mask = Image.open('/root/kj_work/IDM-VTON_old/my_tryon_test_data/mask.png')

    from my_get_pose import InferenceAction
    g_pose = InferenceAction('/root/kj_work/IDM-VTON','/group_share/model/IDM-VTON/densepose')
    pose = g_pose.execute(img_o.copy())
    pose = Image.fromarray(pose)
    # pose = Image.open('/root/kj_work/IDM-VTON_old/my_tryon_test_data/pose.jpg')
    
    # print('img:',img.split())
    # print('cloth:',cloth.split())
    # print('mask:',mask.split())
    # print('pose:',pose.split())
    # exit()
    new = TO.tryon(p1, p2, pose,  cloth, img_o, agnostic)
    # print(3)
    new.save('/root/kj_work/idm_output/new3.jpg')
    # print(4)


# python apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml \
# /root/kj_work/IDM-VTON_old/local_directory/models--yisol--IDM-VTON/snapshots/585a32e74aee241cbc0d0cc3ab21392ca58c916a/densepose/model_final_162be9.pkl \
# /root/kj_work/IDM-VTON_old/my_tryon_test_data dp_segm -v