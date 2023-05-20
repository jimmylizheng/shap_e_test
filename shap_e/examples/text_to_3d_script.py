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

def get_gpu_utilization():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'])
    gpu_util = re.findall(r'\d+', output.decode('utf-8'))
    return int(gpu_util[0])

def get_volatile_gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,nounits,noheader'])
    memory_info = re.findall(r'\d+', output.decode('utf-8'))
    memory_total = int(memory_info[0])
    memory_free = int(memory_info[1])
    memory_used = memory_total - memory_free
    return memory_used

def get_ecc_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.ecc.errors', '--format=csv,nounits,noheader'])
    ecc_memory = re.findall(r'\d+', output.decode('utf-8'))
    return int(ecc_memory[0])

def get_gpu_power_consumption():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,nounits,noheader'])
    power_draw = re.findall(r'\d+\.?\d*', output.decode('utf-8'))
    return float(power_draw[0])
    
def plot_measurement(data, x_label='Time (s)', y_label='Memory Usage (MiB)', title='GPU Memory Usage'):
    """
    Plot the measurement graph.
    """
    timestamps = [t for t, _ in data]
    measured_val = [m for _, m in data]

    plt.plot(timestamps, measured_val)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
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
        self.util_data = []
        self.vol_mem_usage_data = []
        self.power_data = []
        # self.ecc_mem_data = []
        self.start_time = time.time()
        self.interval = interval
        # Create and start the monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_memory)
        self.monitor_thread.start()
        print("Start GPU Moniter")
        
    def monitor_memory(self):
        while True:
            memory_usage = get_gpu_memory_usage()
            util_mem_usage = get_gpu_utilization()
            vol_mem_usage = get_volatile_gpu_memory()
            power_usage=get_gpu_power_consumption()
            # gcc_mem_usage = get_ecc_memory()
            if memory_usage is not None:
                current_time = time.time() - self.start_time
                self.memory_usage_data.append((current_time, memory_usage))
                self.util_data.append((current_time, util_mem_usage))
                self.vol_mem_usage_data.append((current_time, vol_mem_usage))
                self.power_data.append((current_time, power_usage))
                # self.ecc_mem_data.append((current_time, gcc_mem_usage))
                # print(f'Time: {current_time:.2f}s, Memory Usage: {memory_usage} bytes')
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
        
    def mem_plot(self, mode='mem'):
        if mode=='mem':
            plot_measurement(self.memory_usage_data)
        elif mode=='util':
            plot_measurement(self.util_data,'Time (s)','GPU Utilization (%)','GPU Utilization')
        elif mode=='vol':
            plot_measurement(self.vol_mem_usage_data)
        elif mode=='power':
            plot_measurement(self.power_data,'Time (s)','GPU Power Consumption (W)','GPU Power Consumption')
        # elif mode=='ecc':
            # plot_memory_usage(self.ecc_mem_data)

def main():
    gpu_moniter=GPU_moniter(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gpu_memory = get_gpu_memory_usage()
    
    # model parameter counts added inside load_model function
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    
    old_gpu_memory=gpu_memory
    gpu_memory = get_gpu_memory_usage()
    model_gpu_memory = gpu_memory-old_gpu_memory
    print(f"GPU Memory Usage for Loading Model: {model_gpu_memory} MiB")
    print(f"Total GPU Memory Usage before diffusion: {gpu_memory} MiB")
    
    print("start timing deffusion process")
    start_time=time.time()
    
    batch_size = 1
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
    
    print(f"Total GPU Memory Usage before rendering: {gpu_memory} MiB")
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
    print("Total GPU Memory Usage")
    gpu_moniter.mem_plot()
    print("Util GPU Memory Usage")
    gpu_moniter.mem_plot('util')
    print("Volatile GPU Memory Usage")
    gpu_moniter.mem_plot('vol')
    print("GPU Power Consumption")
    gpu_moniter.mem_plot('power')
    
    # Example of saving the latents as meshes.
    # for i, latent in enumerate(latents):
    #     t = decode_latent_mesh(xm, latent).tri_mesh()
    #     with open(f'example_mesh_{i}.ply', 'wb') as f:
    #         t.write_ply(f)
    #     with open(f'example_mesh_{i}.obj', 'w') as f:
    #         t.write_obj(f)
    
if __name__=="__main__":
    main()