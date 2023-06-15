import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image

import time
import psutil
import subprocess
import re
import matplotlib.pyplot as plt
import threading
from IPython.display import display, Image
import io
import json
import os
import urllib.request
from urllib.error import HTTPError
import tempfile

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

def main():
    gpu_mode=False
    timing_mode=False
    save_fig=False
    if gpu_mode:
        gpu_moniter=GPU_moniter(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if gpu_mode:
        gpu_memory = get_gpu_memory_usage()
    
    xm = load_model('decoder', device=device)
    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    
    # Specify the path to the JSON file
    file_path = './coco_data/captions_val2014_fakecap_results.json'
    img_file_path = './coco_data/captions_val2014.json'
    # Open the JSON file and load its contents as a dictionary
    with open(file_path, 'r') as file:
        coco_data = json.load(file)
    with open(img_file_path, 'r') as file:
        coco_img_data = json.load(file)
    
    data_count=0
    time_record={}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Temporary directory created:", temp_dir)
        for data in coco_img_data['images']:
            if data_count>=100:
                diffusion_latency=0
                rendering_latency=0
                for key in time_record:
                    # print(key)
                    # print(time_record[key])
                    time_record[key]['diffusion_latency']=time_record[key]['diffusion_end']-time_record[key]['diffusion_begin']
                    diffusion_latency+=time_record[key]['diffusion_latency']
                    time_record[key]['rendering_latency']=time_record[key]['rendering_end']-time_record[key]['rendering_begin']
                    rendering_latency+=time_record[key]['rendering_latency']
                diffusion_latency=diffusion_latency/data_count
                rendering_latency=rendering_latency/data_count
                outfile = open("./data_result/shap_e_img_decoder_nerf.txt", "a")
                print(time_record, file=outfile)
                print(f"diffusion_latency={diffusion_latency}", file=outfile)
                print(f"renderinging_latency={rendering_latency}", file=outfile)
                outfile.close()
                break
        
            data_count+=1
            batch_size = 1
            guidance_scale = 3.0
            
            try:
                # time_record['current_id']=data['id']
                img_url=data['url']
                filename = os.path.basename(img_url)
                image_path = os.path.join(temp_dir, filename)
                urllib.request.urlretrieve(img_url, image_path)
                # img = Image.open(image_path)

                # To get the best result, you should remove the background and show only the object of interest to the model.
                # image = load_image("./shap_e/examples/example_data/corgi.png")
                image = load_image(image_path)
                
                if gpu_mode:
                    old_gpu_memory=gpu_memory
                    gpu_memory = get_gpu_memory_usage()
                    model_gpu_memory = gpu_memory-old_gpu_memory
                    print(f"GPU Memory Usage for Loading Model: {model_gpu_memory} MiB")
                    print(f"Total GPU Memory Usage before diffusion: {gpu_memory} MiB")
                
                if timing_mode:
                    print("start timing deffusion process")
                    start_time=time.time()
                    
                time_record[data['id']]={}
                time_record[data['id']]['diffusion_begin']=time.time()
                
                latents = sample_latents(
                    batch_size=batch_size,
                    model=model,
                    diffusion=diffusion,
                    guidance_scale=guidance_scale,
                    model_kwargs=dict(images=[image] * batch_size),
                    progress=True,
                    clip_denoised=True,
                    use_fp16=True,
                    use_karras=True,
                    karras_steps=64,
                    sigma_min=1e-3,
                    sigma_max=160,
                    s_churn=0,
                )
                time_record[data['id']]['diffusion_end']=time.time()
                
                if timing_mode:
                    print("end timing deffusion process")
                    end_time=time.time()
                    duration=end_time-start_time
                    print(f"runtime for diffusion process: {duration} seconds")
                
                if gpu_mode:
                    gpu_memory = get_gpu_memory_usage()
                    diffusion_gpu_memory = gpu_memory - model_gpu_memory
                    print(f"GPU Memory Usage for Diffusion: {diffusion_gpu_memory} MiB")
                    print(f"Total GPU Memory Usage before rendering: {gpu_memory} MiB")
                
                if timing_mode:
                    print("start timing rendering process")
                    start_time=time.time()
                
                render_mode = 'nerf' # you can change this to 'stf' for mesh rendering
                size = 64 # this is the size of the renders; higher values take longer to render.
                
                time_record[data['id']]['rendering_begin']=time.time()
                cameras = create_pan_cameras(size, device)
                for i, latent in enumerate(latents):
                    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
                    # display(gif_widget(images))
                time_record[data['id']]['rendering_end']=time.time()
                
                # for image in images:
                #     data = io.BytesIO()
                #     image.save(data, format='PNG')
                #     display(Image(data=data.getvalue()))
                if save_fig:
                    for i, image in enumerate(images):
                        filename = f"shap_e_output_fig_{i}.png"
                        image.save(filename, format='PNG')
                    
                if timing_mode:
                    print("end timing rendering process")
                    end_time=time.time()
                    duration=end_time-start_time
                    print(f"runtime for rendering process: {duration} seconds")
                
                if gpu_mode:
                    old_gpu_memory=gpu_memory
                    gpu_memory = get_gpu_memory_usage()
                    rendering_gpu_memory=gpu_memory-old_gpu_memory
                    print(f"GPU Memory Usage for Rendering: {rendering_gpu_memory} MiB")
                    print(f"Total GPU Memory Usage: {gpu_memory} MiB")
                
                if gpu_mode:
                    gpu_moniter.end_monitor()
                    print("Total GPU Memory Usage")
                    gpu_moniter.mem_plot()
                    print("Util GPU Memory Usage")
                    gpu_moniter.mem_plot('util')
                    print("Volatile GPU Memory Usage")
                    gpu_moniter.mem_plot('vol')
                    print("GPU Power Consumption")
                    gpu_moniter.mem_plot('power')
            except HTTPError as e:
                print("An HTTP error occurred:", e.code, e.reason)
            except Exception as e:
                print("An error occurred:", str(e))
    
if __name__=="__main__":
    main()