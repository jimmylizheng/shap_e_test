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
    # Open the file in write mode
    xm_name = 'decoder'
    model_name='image300M'
    # sys.stdout = open('load-text300M-trans-mix.txt', 'a')
    init_t=time.time()
    print(f"b{get_gpu_memory_usage()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    xm = load_model(xm_name, device=device)
    
    print(f"d{get_gpu_memory_usage()}")
    print(f"d{time.time()-init_t}")

    model = load_model(model_name, device=device)

    diffusion = diffusion_from_config(load_config('diffusion'))
    
    print(f"e{get_gpu_memory_usage()}")

    print(f"e{time.time()-init_t}")
    # # Remember to close the file to ensure everything is saved
    # sys.stdout.close()

    # # Reset the stdout to its default value (the console)
    # sys.stdout = sys.__stdout__
    

if __name__ == "__main__":
    main()