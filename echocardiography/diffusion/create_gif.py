"""
Create a gif fro a set of images in a folder
"""
import os
import imageio
import argparse
import numpy as np
from tqdm import tqdm

def create_gif(image_folder, gif_name):
    images = []
    frames = np.arange(999, -1, -1)
    for im in frames:
        im_path = os.path.join(image_folder, f'x0_{im}.png')
        images.append(imageio.imread(im_path))
    imageio.mimsave(f'{gif_name}.gif', images, duration=0.01)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for creating gif')
    parser.add_argument('--image_folder', type=str, help='Folder containing images to create gif from')
    parser.add_argument('--gif_name', type=str, help='Name of gif to save')
    args = parser.parse_args()
    
    create_gif(args.image_folder, args.gif_name)