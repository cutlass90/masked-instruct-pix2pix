import torch
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image
from pipeline_stable_diffusion_masked_instruct_pix2pix import StableDiffusionMaskedInstructPix2PixPipeline
from imageio.v2 import imread
import numpy as np

prompt = "add a knight helmet"
device = 'cuda:0'
model_id = "timbrooks/instruct-pix2pix"
STEPS = 20
image = Image.open('data/nazar person.png')
mask = (imread('data/helmet_mask.png')[:,:,0]/255).astype(np.float16)

pipe = StableDiffusionMaskedInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


for i in range(10):
    result_image = pipe(prompt, image=image, num_inference_steps=STEPS,  guidance_scale=14, mask=mask).images[0]
    result_image.save('data/'+prompt+f'{i}.png')
