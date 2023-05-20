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
import matplotlib.pyplot as plt
import threading

def get_gpu_memory_usage():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    memory_used = re.findall(r'\d+', output.decode('utf-8'))
    return int(memory_used[0])

def plot_memory_usage(memory_usage_data):
    """
    Plot the memory usage graph.
    """
    timestamps = [t for t, _ in memory_usage_data]
    memory_usages = [m for _, m in memory_usage_data]

    plt.plot(timestamps, memory_usages)
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (bytes)')
    plt.title('GPU Memory Usage')
    plt.grid(True)
    plt.show()
    
class GPU_moniter:
    """
    Monitor the GPU memory usage every 'interval' seconds until the program completes.
    """
    def __init__(self, interval=1):
        """Initialize GPU_moniter."""
        self.stop_flag=False
        self.memory_usage_data = []
        self.start_time = time.time()
        self.interval = interval
        # Create and start the monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_memory)
        self.monitor_thread.start()
        print("Start GPU Moniter")
        
    def monitor_memory(self):
        while True:
            memory_usage = get_gpu_memory_usage()
            if memory_usage is not None:
                current_time = time.time() - self.start_time
                self.memory_usage_data.append((current_time, memory_usage))
                print(f'Time: {current_time:.2f}s, Memory Usage: {memory_usage} bytes')
            else:
                print('Failed to retrieve GPU memory usage.')

            # Check if the program has completed
            if self.stop_flag:
                break
            time.sleep(self.interval)
    
    def end_monitor(self):
        self.stop_flag=True
        
        # Wait for the monitoring thread to complete
        self.monitor_thread.join()

        plot_memory_usage(self.memory_usage_data)
    
def monitor_gpu_memory_usage(interval=1):
    """
    Monitor the GPU memory usage every 'interval' seconds until the program completes.
    """
    memory_usage_data = []
    start_time = time.time()

    def monitor_memory():
        nonlocal memory_usage_data
        nonlocal start_time

        while True:
            memory_usage = get_gpu_memory_usage()
            if memory_usage is not None:
                current_time = time.time() - start_time
                memory_usage_data.append((current_time, memory_usage))
                # print(f'Time: {current_time:.2f}s, Memory Usage: {memory_usage} bytes')
            else:
                print('Failed to retrieve GPU memory usage.')

            # Check if the program has completed
            # if stop_flag:
            #     break
            # time.sleep(interval)

    # Create and start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.start()

    # Wait for the monitoring thread to complete
    monitor_thread.join()

    plot_memory_usage(memory_usage_data)

def main():
    gpu_moniter=GPU_moniter(1)
    
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
    
    gpu_moniter.end_monitor()
    
    # Example of saving the latents as meshes.
    # for i, latent in enumerate(latents):
    #     t = decode_latent_mesh(xm, latent).tri_mesh()
    #     with open(f'example_mesh_{i}.ply', 'wb') as f:
    #         t.write_ply(f)
    #     with open(f'example_mesh_{i}.obj', 'w') as f:
    #         t.write_obj(f)
    
if __name__=="__main__":
    main()