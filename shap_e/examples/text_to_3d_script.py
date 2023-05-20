import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh

import time
import psutil
import subprocess
import re

def get_gpu_memory_usage():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    memory_used = re.findall(r'\d+', output.decode('utf-8'))
    return int(memory_used[0])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gpu_memory = get_gpu_memory_usage()
    
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    
    old_gpu_memory=gpu_memory
    gpu_memory = get_gpu_memory_usage()
    model_gpu_memory = gpu_memory-old_gpu_memory
    print(f"GPU Memory Usage for Loading Model: {model_gpu_memory} MiB")
    
    print("start timing deffusion process")
    start_time=time.time()
    
    batch_size = 4
    guidance_scale = 15.0
    prompt = "a shark"

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    
    print("end timing deffusion process")
    end_time=time.time()
    duration=end_time-start_time
    print(f"runtime for diffusion process: {duration} seconds")
    
    gpu_memory = get_gpu_memory_usage()
    diffusion_gpu_memory = gpu_memory - model_gpu_memory
    print(f"GPU Memory Usage for Diffusion: {diffusion_gpu_memory} MiB")
    
    print("start timing rendering process")
    start_time=time.time()
    
    render_mode = 'nerf' # you can change this to 'stf'
    size = 64 # this is the size of the renders; higher values take longer to render.

    cameras = create_pan_cameras(size, device)
    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        # display(gif_widget(images))
        
    print("end timing rendering process")
    end_time=time.time()
    duration=end_time-start_time
    print(f"runtime for rendering process: {duration} seconds")
    
    old_gpu_memory=gpu_memory
    gpu_memory = get_gpu_memory_usage()
    rendering_gpu_memory=gpu_memory-old_gpu_memory
    print(f"GPU Memory Usage for Rendering: {rendering_gpu_memory} MiB")
    print(f"Total GPU Memory Usage: {gpu_memory} MiB")
    
    # Example of saving the latents as meshes.
    # for i, latent in enumerate(latents):
    #     t = decode_latent_mesh(xm, latent).tri_mesh()
    #     with open(f'example_mesh_{i}.ply', 'wb') as f:
    #         t.write_ply(f)
    #     with open(f'example_mesh_{i}.obj', 'w') as f:
    #         t.write_obj(f)
    
if __name__=="__main__":
    main()