import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from registrator import Registrator
import math
import tifffile as tiff
from pathlib import Path
from scipy.ndimage import zoom
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


### padding for square-shaped image generation
def fit_to_square(img):
    y, x, _ = img.shape

    # fit to even numbers
    if y%2 != 0:
        img = np.pad(img, [(1, 0), (0, 0), (0, 0)], mode='maximum')
    elif x%2 != 0:
        img = np.pad(img, [(0, 0), (1, 0), (0, 0)], mode='maximum')

    # padding
    pad = abs(y-x) // 2
    img = np.pad(img, ((0, 0), (pad, pad), (0, 0)), mode='maximum') if y > x else np.pad(img, ((pad, pad), (0, 0), (0, 0)), mode='maximum')

    return img

### Pen masking
def masking(img, color='blue'):
    if color == 'none':
        return img
    elif color == 'blue' or color == 'black':
        mask = img[..., 0] < 200
    elif color == 'red':
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = img_hsv[..., 1] > 200
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)
    opening_dil = cv2.dilate(opening, kernel ,iterations=7)
    mask_new = opening_dil
    img[mask_new==True] = (255, 255, 255)
    return img


### image simplifying
def simplifying(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thres = (img < 245).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((6, 6), np.uint8)

    closing = cv2.morphologyEx(thres.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=3)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2, iterations=1)

    return closing

# Load saved image

datapath = Path('./220208_rotate_tiff')
folders = sorted(datapath.glob('*'))
braf_folders = sorted(datapath.glob('*BRAF*'))
hne_folders = [x for x in folders if x not in braf_folders]


braf_large_list = [list(x.glob('*Extended.tif'))[0] for x in braf_folders][40:]
hne_large_list = [list(x.glob('*Extended.tif'))[0] for x in hne_folders][40:]

df_masking = pd.read_csv('sample_labelling.csv', index_col=0)
for i, paths in enumerate(tqdm(zip(braf_large_list, hne_large_list), total=len(braf_large_list))):
    sample_no = paths[1].parent.name
    pencolor = list(df_masking[df_masking['name']==sample_no]['pencolor'])[0]

    braf = np.array(tiff.imread(paths[0]))
    hne = np.array(tiff.imread(paths[1]))
    braf = fit_to_square(braf)
    hne = fit_to_square(hne)
    braf_small = cv2.resize(braf, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
    hne_small = cv2.resize(hne, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
    hne_small = masking(hne_small, color=pencolor)
    braf_small = simplifying(braf_small)
    hne_small = simplifying(hne_small)
    braf_small = np.expand_dims(braf_small, 2)
    hne_small = np.expand_dims(hne_small, 2)
    
    lr = 0.01
    loss = "mse"
    num_epochs = 400
    registrator = Registrator(
        lr=lr, loss=loss, in_chans=1, num_epochs=num_epochs, device=7
    )
    registrator.initialize()
    # out = registrator.train(braf_mask[..., None], hne_mask[..., None])
    out = registrator.train(hne_small, braf_small)
    
    hne_registrated = registrator.infer(hne / 255)
    tiff.imwrite(paths[0].parent / f"{paths[0].stem}_registered.tif", braf)
    tiff.imwrite(paths[1].parent / f"{paths[1].stem}_registered.tif", hne_registrated)
