# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os
from PIL import Image 
#
#train_imgs = h5py.File('./input_data/train_images.hdf5', 'r')
#for i in range(0,10):
#    print( "now showing image number " , i)
#    
#    fig = plt.figure(figsize=[12, 6])
#    ax1 , ax2  = fig.subplots(1,2)
#    
#    ax1.imshow(train_imgs['target_masks'][i][...], origin='upper', cmap='Greys_r')
#    ax2.imshow(train_imgs['input_images'][i][...], origin='upper', cmap='Greys_r')
#    
#    plt.show()
#    


 #Use scikit-image template matching to extract crater locations.  Only search for craters with r >= 3 pixels.

train_imgs = h5py.File('./input_data/train_images.hdf5', 'r')

crater=pd.HDFStore('./input_data/train_craters.hdf5', 'r')

for i in range(0,100):
#    extracted_rings = tmt.template_match_t(preds[i].copy(), minrad=1.)
    fig = plt.figure(figsize=[16, 16])
    [[ax1, ax2], [ax3, ax4]] = fig.subplots(2,2)
    ax1.imshow(train_imgs['input_images'][i][...], origin='upper', cmap='Greys_r')
    ax2.imshow(train_imgs['target_masks'][i][...], origin='upper', cmap='Greys_r', vmin=0, vmax=1)
#    ax3.imshow(preds[i], origin='upper', cmap='Greys_r', vmin=0, vmax=1)
#    ax4.imshow(train_imgs['input_images'][i][...], origin='upper', cmap="Greys_r")
#    for x, y, r in extracted_rings:
#        circle = plt.Circle((x, y), r, color='blue', fill=False, linewidth=2, alpha=0.5)
#        ax4.add_artist(circle)
    ax1.set_title('Venus DEM Image')
    ax2.set_title('Ground-Truth Target Mask')
    ax3.set_title('CNN Predictions')
    ax4.set_title('Post-CNN Craters')
    plt.show()