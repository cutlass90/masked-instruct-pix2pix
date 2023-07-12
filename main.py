import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

prompt = "add a knight helmet on his head"
device = 'cuda:0'
mask = torch.zeros([1, 1, 64, 64]).to(device).half()
mask[:,:,:, :32] = 1

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
image = Image.open('data/nazar person.png')
latents = torch.randn([1, 4, 64, 64]).to(device).half()
generator = torch.Generator(device=device).manual_seed(43)

with torch.no_grad():
    encoded_latent = pipe.prepare_image_latents(
                pipe.image_processor.preprocess(image),
                1,
                1,
                torch.float16,
                device,
                False,
                generator,
            )



denoised_latent = pipe(prompt, image=image, num_inference_steps=15, latents=latents, generator=generator, output_type="latent", guidance_scale=10).images / pipe.vae.config.scaling_factor
with torch.no_grad():
    result_tensor = pipe.vae.decode(denoised_latent*mask+encoded_latent*(1-mask), return_dict=False)[0].detach()
    result_pil = pipe.image_processor.postprocess(result_tensor, output_type="pil", do_denormalize=[True])[0]
result_pil.save('data/'+prompt+'.png')