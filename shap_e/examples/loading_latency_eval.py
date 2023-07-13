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
from IPython.display import display, Image
import io
import sys
import json


def get_gpu_memory_usage():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    memory_used = re.findall(r'\d+', output.decode('utf-8'))
    return int(memory_used[0])

def main():
    # device = torch.device("cuda")
    # # Get the amount of total memory in bytes
    # total_memory = torch.cuda.get_device_properties(device).total_memory
    # print(total_memory)
    # return
    
    # Open the file in write mode
    xm_name = 'transmitter'
    model_name='text300M'
    sys.stdout = open('load-text300M.txt', 'a')
    # init_t=time.time()
    # print(f"b{get_gpu_memory_usage()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_t=time.time()
    # init_g=get_gpu_memory_usage()
    model = load_model(model_name, device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    # xm = load_model(xm_name, device=device)
    # print(f"e{get_gpu_memory_usage()-init_g}")
    print(f"{time.time()-init_t}")
    
    # torch.save(xm, 'decoder_all.pt')
    # init_t=time.time()
    # init_g=get_gpu_memory_usage()
    # print(f"d{get_gpu_memory_usage()}")
    # print(f"d{time.time()-init_t}")
    # xm = load_model(xm_name, device=device)
    # model = load_model(model_name, device=device)
    # model1 = torch.load('img300_all.pt')

    # diffusion = diffusion_from_config(load_config('diffusion'))
    # print(f"e{get_gpu_memory_usage()-init_g}")
    # print(f"e{time.time()-init_t}")
    
    # init_t=time.time()
    # init_g=get_gpu_memory_usage()
    # # print(f"d{get_gpu_memory_usage()}")
    # print(f"d{time.time()-init_t}")

    # # model = load_model(model_name, device=device)
    # # model2 = torch.load('img300_all.pt')

    # # diffusion = diffusion_from_config(load_config('diffusion'))
    # print(f"e{get_gpu_memory_usage()-init_g}")
    # print(f"e{time.time()-init_t}")
    
    # print(f"total: {get_gpu_memory_usage()}")
    # Remember to close the file to ensure everything is saved
    sys.stdout.close()

    # Reset the stdout to its default value (the console)
    sys.stdout = sys.__stdout__
    

if __name__ == "__main__":
    main()