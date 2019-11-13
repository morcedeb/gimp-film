#!/usr/bin/env python
#
# -------------------------------------------------------------------------------------
#
# Copyright (c) 2019, Marco Calemme
# All rights reserved.
#
# -------------------------------------------------------------------------------------
#

from gimpfu import *
import numpy as np
import skimage.color as color
import skimage.morphology as morpho


DEBUG_MESSAGES = True

def gimp_log(text):
    if DEBUG_MESSAGES:
        pdb.gimp_message(text)


def channelData(layer):
    """
    Returns NP array (N,bpp) (single vector ot triplets)
    """
    # region=layer.get_pixel_rgn(0, 0, layer.width,layer.height)
    # region allows acces to pixel values as pixel = region[x,y] in the form '\x1eE\x86'
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :] # Take whole layer
    bpp = region.bpp
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(len(pixChars)/bpp, bpp)

def createMaskLayer(image, layer_name, raster_img):
    """
    Creates layer into image from result
    """
    # layer parameters
    layertype = 1 #RGBA
    opacity = 100
    mode = 0
    position = 0

    # create new layer and assign bytes
    rl = pdb.gimp_layer_new(image, image.width, image.height, layertype, layer_name, opacity, mode)
    rlBytes = np.uint8(raster_img).tobytes()
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    pdb.gimp_image_insert_layer(image, rl, None, position)
    gimp.displays_flush()

# Thanks xenoid: https://stackoverflow.com/questions/47536186/gimp-python-plugin-gimp-image-as-numpy-array


def detect_dust(image, drawable, isnegative=True, sensitivity=85, spot_size=9):
    """
    Detects dust particles.
    """
    # image = gimp.image_list()[0]
    sensitivity = 1.0*sensitivity/100
    spot_size = int(spot_size)

    H = pdb.gimp_image_height(image)
    W = pdb.gimp_image_width(image)
    # first let's get the pixel values
    layer = pdb.gimp_image_get_active_layer(image)

    raster_image = channelData(layer).reshape([H, W, 3])
    bpp = 8 #TODO get this
    MAX = 2**bpp -1

    # to grayscale
    raster_image_gray = color.rgb2gray(raster_image)
    if isnegative:
        mask_level = raster_image_gray > sensitivity
        # white top hat to avoid uniform burned areas
        strel = morpho.square(spot_size)
        mask_th = morpho.white_tophat(raster_image_gray, selem=strel)
        # take only (very) outliers
        mask_th = mask_th > mask_th.mean() + 3*mask_th.std()
    else: # positive
        mask_level = raster_image_gray < 1-sensitivity
        # black top hat to avoid uniform dark areas
        strel = morpho.square(spot_size)
        mask_th = morpho.black_tophat(raster_image_gray, selem=strel)
        # take only (very) outliers
        mask_th = mask_th > mask_th.mean() + 3*mask_th.std()

    strel = morpho.disk(1)
    mask_th = morpho.opening(mask_th, selem=strel)

    # mask composition
    mask = mask_th & mask_level
    
    # morphological dilation
    strel = morpho.disk(3)
    mask = morpho.dilation(mask, selem=strel)

    # layer composition
    mask_list = [mask*MAX for _ in range(4)]
    dust_mask_rgba = np.stack(mask_list, axis=2)

    createMaskLayer(image, "dust", dust_mask_rgba)


register(
    "detect_dust",
    "Detect dust particles.",
    "Detect dust particles.",
    "Marco Calemme",
    "Marco Calemme - Copyright 2019",
    "2019",
    "<Image>/Filters/my plugins/Detect dust particles",
    "",
    [
        (PF_BOOL, "isnegative", "Negative", True),
        (PF_SLIDER, "sensitivity", "Sensitivity", 85, (75,95,5)),
        (PF_SLIDER, "spot_size", "Spot size", 9, (5,30,2))
    ],
    [],
    detect_dust
    )

main()
