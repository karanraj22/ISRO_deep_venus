#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 01:39:47 2018

@author: karan
"""


from __future__ import absolute_import, division, print_function
from keras.models import Model
from keras.layers.core import Dropout, Reshape
from keras.regularizers import l2

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.models import load_model
import numpy as np
import h5py
import get_unique_craters as guc
import sys


import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os
from PIL import Image 
import numpy as np
import sys
import utils.template_match_target as tmt
import utils.processing as proc
import utils.transform as trf
# Crater Parameters
CP = {}

# Image width/height, assuming square images.
CP['dim'] = 256

# Data type - train, dev, test
CP['datatype'] = 'train'

# Number of images to extract craters from
CP['n_imgs'] = 20
CP['start_of_images'] = 0

# Hyperparameters
#Changed the learning rates @Edit by - Karanraj Singh Saini

CP['llt2'] = 3   # D_{L,L} from Silburt et. al (2017)
CP['rt2'] = 3    # D_{R} from Silburt et. al (2017)

# Location of model to generate predictions (if they don't exist yet)
CP['dir_model'] = '/home/karan/deepmoon/models/model_keras2.h5'

# Location of where hdf5 data images are stored
CP['dir_data'] = '/home/karan/deepvenus_excel_mask/input_data/%s_images.hdf5' % CP['datatype']

# Location of where model predictions are/will be stored
CP['dir_preds'] = 'catalogues/%s_preds_n%d_start%d.hdf5' % (CP['datatype'],
                                                    CP['n_imgs'],
                                                    CP['start_of_images'] )

# Location of where final unique crater distribution will be stored
CP['dir_result'] = 'catalogues/%s_craterdist.npy' % (CP['datatype'])

#!/usr/bin/env python
"""Input Image Dataset Generator

Script for generating input datasets from Lunar global digital elevation maps 
(DEMs) and crater catalogs.

This script is designed to use the LRO-Kaguya DEM and a combination of the
LOLA-LROC 5 - 20 km and Head et al. 2010 >=20 km crater catalogs.  It
generates a randomized set of small (projection-corrected) images and
corresponding crater targets.  The input and target image sets are stored as
hdf5 files.  The longitude and latitude limits of each image is included in the
input set file, and tables of the craters in each image are stored in a
separate Pandas HDFStore hdf5 file.

The script's parameters are located under the Global Variables.  We recommend
making a copy of this script when generating a dataset.

MPI4py can be used to generate multiple hdf5 files simultaneously - each thread
writes `amt` number of images to its own file.
"""

########## Imports ##########

# Python 2.7 compatibility.
from __future__ import absolute_import, division, print_function

from PIL import Image
import input_data_gen as igen
import time

########## Global Variables ##########

# Use MPI4py?  Set this to False if it's not supposed by the system.
use_mpi4py = False

# Source image path.
source_image_path = "/home/karan/gdal_env/outresize.png"

# LROC crater catalog csv path.
#lroc_csv_path = "./catalogues/LROCCraters.csv"

# Head et al. catalog csv path.
head_csv_path = "./catalogues/venuscrater.csv"

# Output filepath and file header.  Eg. if outhead = "./input_data/train",
# files will have extension "./out/train_inputs.hdf5" and
# "./out/train_targets.hdf5"
outhead = "./input_data/train"

# Number of images to make (if using MPI4py, number of image per thread to
# make).
amt = 100

# Range of image widths, in pixels, to crop from source image (input images
# will be scaled down to ilen). For Orthogonal projection, larger images are
# distorted at their edges, so there is some trade-off between ensuring images
# have minimal distortion, and including the largest craters in the image.
rawlen_range = [500, 10000]

# Distribution to sample from rawlen_range - "uniform" for uniform, and "log"
# for loguniform.
rawlen_dist = 'log'

# Size of input images.
ilen = 256

# Size of target images.
tglen = 256

# [Min long, max long, min lat, max lat] dimensions of source image.
source_cdim = [-180., 180., -90., 90.]

# [Min long, max long, min lat, max lat] dimensions of the region of the source
# to use when randomly cropping.  Used to distinguish training from test sets.
sub_cdim = [-180., 180., -90., 90.]

# Minimum pixel diameter of craters to include in in the target.
minpix = 1.

# Radius of the world in km (1737.4 for Moon).
R_km = 6052

### Target mask arguments. ###

# If True, truncate mask where image has padding.
truncate = True

# If rings = True, thickness of ring in pixels.
ringwidth = 1

# If True, script prints out the image it's currently working on.
verbose = True
def get_model_preds(CP):
    """Reads in or generates model predictions.

    Parameters
    ----------
    CP : dict
        Containins directory locations for loading data and storing
        predictions.

    Returns
    -------
    craters : h5py
        Model predictions.
    """
    n_imgs, dtype = CP['n_imgs'], CP['datatype']

    data = h5py.File(CP['dir_data'], 'r')

    Data = {
        dtype: [data['input_images'][:n_imgs].astype('float32'),
                data['target_masks'][:n_imgs].astype('float32')]
    }
    data.close()
    proc.preprocess(Data)

    model = load_model(CP['dir_model'])
    preds = model.predict(Data[dtype][0])

    # save
    h5f = h5py.File(CP['dir_preds'], 'w')
    h5f.create_dataset(dtype, data=preds)
    print("Successfully generated and saved model predictions.")
    return preds

def AddPlateCarree_XY(craters, imgdim, cdim=[-180., 180., -90., 90.], 
                      origin="upper"):
    """Adds x and y pixel locations to craters dataframe.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater info
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels
    cdim : list-like, optional
        Coordinate limits (x_min, x_max, y_min, y_max) of image.  Default is
        [-180., 180., -90., 90.].
    origin : "upper" or "lower", optional
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.
    """
    x, y = trf.coord2pix(craters["Long"].as_matrix(),
                         craters["Lat"].as_matrix(),
                         cdim, imgdim, origin=origin)
    craters["x"] = x
    craters["y"] = y
    
def get_model_preds(CP):
    """Reads in or generates model predictions.

    Parameters
    ----------
    CP : dict
        Containins directory locations for loading data and storing
        predictions.

    Returns
    -------
    craters : h5py
        Model predictions.
    """
    n_imgs, dtype = CP['n_imgs'], CP['datatype']

    data = h5py.File(CP['dir_data'], 'r')

    Data = {
        dtype: [data['input_images'][:n_imgs].astype('float32'),
                data['target_masks'][:n_imgs].astype('float32')]
    }
    data.close()
    proc.preprocess(Data)

    model = load_model(CP['dir_model'])
    preds = model.predict(Data[dtype][0])

    # save
    h5f = h5py.File(CP['dir_preds'], 'w')
    h5f.create_dataset(dtype, data=preds)
    print("Successfully generated and saved model predictions.")
    return preds ,Data[dtype]
def get_metrics(data, craters, dim, model, beta=1):
    """Function that prints pertinent metrics at the end of each epoch. 

    Parameters
    ----------
    data : hdf5
        Input images.
    craters : hdf5
        Pandas arrays of human-counted crater data. 
    dim : int
        Dimension of input images (assumes square).
    model : keras model object
        Keras model
    beta : int, optional
        Beta value when calculating F-beta score. Defaults to 1.
    """
    X, Y = data[0], data[1]
    dim =256
    # Get csvs of human-counted craters
    csvs = []
    minrad, maxrad, cutrad, n_csvs = 1, 50, 0.8, len(X)
    diam = 'Diameter (pix)'
    for i in range(n_csvs):
        try:
            csv = craters[proc.get_id(i,4)]
        except:
            csvs.append([-1])
            print ('Skipping iteration number =' ,i  ,' as no crater is available in that area')
            continue
        # remove small/large/half craters
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
        if len(csv) < 1:    # Exclude csvs with few craters
            csvs.append([-1])
        else:
            csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T
            csvs.append(csv_coords)

    # Calculate custom metrics
    print("")
    print("*********Custom Loss*********")
    recall, precision, fscore = [], [], []
    frac_new, frac_new2, maxrad = [], [], []
    err_lo, err_la, err_r = [], [], []
    frac_duplicates = []
#    print(len(csvs[1]))
#    print(len(csvs[2]))
    preds = model.predict(X)
    for i in range(len(csvs)):
        if len(csvs[i]) < 1:
            continue
        
        try:
            if (csvs[i].count(-1) ==1):
                print ('Skipping bcoz csvs ==-1')
                continue
        except:
            print ('out of try block')
      
        (N_match, N_csv, N_detect, maxr,
         elo, ela, er, frac_dupes) = tmt.template_match_t2c(preds[i], csvs[i],
                                                            rmv_oor_csvs=0)
        if N_match > 0:
            p = float(N_match) / float(N_match + (N_detect - N_match))
            r = float(N_match) / float(N_csv)
            f = (1 + beta**2) * (r * p) / (p * beta**2 + r)
            diff = float(N_detect - N_match)
            fn = diff / (float(N_detect) + diff)
            fn2 = diff / (float(N_csv) + diff)
            recall.append(r)
            precision.append(p)
            fscore.append(f)
            frac_new.append(fn)
            frac_new2.append(fn2)
            maxrad.append(maxr)
            err_lo.append(elo)
            err_la.append(ela)
            err_r.append(er)
            frac_duplicates.append(frac_dupes)
        else:
            print("skipping iteration %d,N_csv=%d,N_detect=%d,N_match=%d" %
                  (i, N_csv, N_detect, N_match))

    print("binary XE score = %f" % model.evaluate(X, Y))
    if len(recall) > 0:
        print("mean and std of N_match/N_csv (recall) = %f, %f" %
              (np.mean(recall), np.std(recall)))
        print("""mean and std of N_match/(N_match + (N_detect-N_match))
              (precision) = %f, %f""" % (np.mean(precision), np.std(precision)))
        print("mean and std of F_%d score = %f, %f" %
              (beta, np.mean(fscore), np.std(fscore)))
        print("""mean and std of (N_detect - N_match)/N_detect (fraction
              of craters that are new) = %f, %f""" %
              (np.mean(frac_new), np.std(frac_new)))
        print("""mean and std of (N_detect - N_match)/N_csv (fraction of
              "craters that are new, 2) = %f, %f""" %
              (np.mean(frac_new2), np.std(frac_new2)))
        print("median and IQR fractional longitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_lo), np.percentile(err_lo, 25),
               np.percentile(err_lo, 75)))
        print("median and IQR fractional latitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_la), np.percentile(err_la, 25),
               np.percentile(err_la, 75)))
        print("median and IQR fractional radius diff = %f, 25:%f, 75:%f" %
              (np.median(err_r), np.percentile(err_r, 25),
               np.percentile(err_r, 75)))
        print("mean and std of frac_duplicates: %f, %f" %
              (np.mean(frac_duplicates), np.std(frac_duplicates)))
        print("""mean and std of maximum detected pixel radius in an image =
              %f, %f""" % (np.mean(maxrad), np.std(maxrad)))
        print("""absolute maximum detected pixel radius over all images =
              %f""" % np.max(maxrad))
        print("")

def keys(f):
    return [key for key in f.keys()]
'''             
--------------------------------------------------------------------------------------
'''


preds , data = get_model_preds(CP)


 #Use scikit-image template matching to extract crater locations.  Only search for craters with r >= 3 pixels.
extracted_rings = tmt.template_match_t(preds[6].copy(), minrad=1.)
train_imgs = h5py.File('./input_data/train_images.hdf5', 'r')

crater=pd.HDFStore('./input_data/train_craters.hdf5', 'r')


fig = plt.figure(figsize=[16, 16])
[[ax1, ax2], [ax3, ax4]] = fig.subplots(2,2)
ax1.imshow(train_imgs['input_images'][6][...], origin='upper', cmap='Greys_r', vmin=50, vmax=200)
ax2.imshow(train_imgs['target_masks'][6][...], origin='upper', cmap='Greys_r', vmin=0, vmax=1)
ax3.imshow(preds[6], origin='upper', cmap='Greys_r', vmin=0, vmax=1)
ax4.imshow(train_imgs['input_images'][6][...], origin='upper', cmap="Greys_r")
for x, y, r in extracted_rings:
    circle = plt.Circle((x, y), r, color='blue', fill=False, linewidth=2, alpha=0.5)
    ax4.add_artist(circle)
ax1.set_title('Venus DEM Image')
ax2.set_title('Ground-Truth Target Mask')
ax3.set_title('CNN Predictions')
ax4.set_title('Post-CNN Craters')
plt.show()


#get craters info from csv 

# Get craters.
siz = (92610,30720)

model = load_model(CP['dir_model'])
get_metrics(data,crater,256,model)


