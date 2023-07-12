import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

prompt = "add a knight helmet on his head"
device = 'cuda:0'

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
image = Image.open('data/nazar person.png')
latents = torch.randn([1, 4, 64, 64]).to(device).half()
generator = torch.Generator(device=device).manual_seed(42)
denoised_latent = pipe(prompt, image=image, num_inference_steps=15, latents=latents, generator=generator, output_type="latent")
# denoised_latent = pipe(prompt, image=image, num_inference_steps=15, latents=latents, generator=generator)
result_tensor = pipe.vae.decode(denoised_latent.images / pipe.vae.config.scaling_factor, return_dict=False)[0]
result_pil = pipe.image_processor.postprocess(result_tensor, output_type="pil", do_denormalize=[True])

result_pil.save('data/'+prompt+'.png')