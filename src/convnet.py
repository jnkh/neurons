# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:06:55 2016

@author: tpisano, jnkh
"""
import os, numpy as np, cv2, zipfile
from skimage.external import tifffile #if this doesn't work import tifffile

#%%


if __name__ == '__main__':
    ########################
    ###to run for single contour image pair
    pth = 'path_to_image_zip'#'/home/wanglab/wang/pisano/conv_net/training/shriansh/image_zip_pairs/bl6_20150929_crsI_01_647_70msec_z3um_3hfds_C00_Z1357'
    parsed_im, parsed_mask = parser(pth, size=(200, 200))

    ########################
    ###to run for all files:
    fld = '/home/wanglab/wang/pisano/conv_net/training/shriansh/image_zip_pairs'
    im_zp_pair_pths = [os.path.join(fld, xx) for xx in os.listdir(fld)]
    im_zp_pair_lst =  [parser(xx, size= (200,200)) for xx in im_zp_pair_pths]
    im_zp_pair_lst = [xx for xx in im_zp_pair_lst if xx != None] #remove any failed jobs
    #unpack
    parsed_im = [xxx for xx in im_zp_pair_lst for xxx in xx[0]]
    parsed_mask = [xxx for xx in im_zp_pair_lst for xxx in xx[1]]
    
#%%

def parser(pth, size = (228, 228)):
    '''Function to take image and ROI zip pairs made using ImageJ, generate two lists corresponding to im and masked image.
    Note, many of the image masks will be blank images, particularly with a sparsely labeled section.
    
    Parameters
    -----------
    pth: path to folder containing tif and roi.zip
    size: tuple, x by y output parsed image size
    
    Returns
    -----------
    parsed_im: list of parsed images to appropriate size
    parsed_mask: list of parsed binary mask of filled contours (pixelvalue=1)
    
    '''
    #set pth to zip and load im    
    try:        
        zipfl = [os.path.join(pth, xx) for xx in os.listdir(pth) if 'zip' in xx[-3:]][0]    
        im = tifffile.imread([os.path.join(pth, xx) for xx in os.listdir(pth) if 'tif' in xx[-3:]][0])  
    except IndexError:
        print('missing files for:\n     {}'.format(pth))
        return
    
    #unpack zipfl into a list of roi contours; note convention change ImageJ=x,y, cv2=y,x
    roilst=[]    
    zp=zipfile.ZipFile(zipfl)
    for i in zp.namelist():
        roilst.append(np.fliplr(np.int32(read_roi(zp.open(i))))) 
    
    #create a binary mask of im:    
    mask = np.zeros(im.shape)
    cv2.fillPoly(mask, roilst, 1)
    
    #parse up im and mask
    parsed_im = generate_blocks(im,size) 
    parsed_mask = generate_blocks(mask,size)
    
    print('Completed parsing: {}'.format(pth[pth.rfind('/')+1:]))
    return parsed_im, parsed_mask



def generate_blocks(arr,size):
    tot_size_x = size[0] * (arr.shape[0] / size[0])
    tot_size_y = size[1] * (arr.shape[1] / size[1])
    arr = arr[:tot_size_x,:tot_size_y]
    first_split = np.array_split(arr, arr.shape[0] / size[0], axis = 0)
    ret = []
    for s in first_split:
        for x in np.array_split(s,s.shape[1] / size[1],axis=1):
            ret.append(x)
    return ret


#### Author Julian Kates-Harbeck####

def normalize_arr(arrs,new_shape):
    max_val = max([np.max(a) for a in arrs])
    for i in range(len(arrs)):
        arrs[i] = arrs[i]*1.0/max_val
        arrs[i] = np.reshape(arrs[i],new_shape)
    return np.stack(arrs)

def normalize(X,y,new_shape):
    return normalize_arr(X,new_shape),normalize_arr(y,new_shape)



#%%
# Copyright: Luis Pedro Coelho <luis@luispedro.org>, 2012
# License: MIT

def read_roi(fileobj):
    '''
    points = read_roi(fileobj)
    Read ImageJ's ROI format
    '''
# This is based on:
# http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
# http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html


    SPLINE_FIT = 1
    DOUBLE_HEADED = 2
    OUTLINE = 4
    OVERLAY_LABELS = 8
    OVERLAY_NAMES = 16
    OVERLAY_BACKGROUNDS = 32
    OVERLAY_BOLD = 64
    SUB_PIXEL_RESOLUTION = 128
    DRAW_OFFSET = 256


    pos = [4]
    def get8():
        pos[0] += 1
        s = fileobj.read(1)
        if not s:
            raise IOError('readroi: Unexpected EOF')
        return ord(s)

    def get16():
        b0 = get8()
        b1 = get8()
        return (b0 << 8) | b1

    def get32():
        s0 = get16()
        s1 = get16()
        return (s0 << 16) | s1

    def getfloat():
        v = np.int32(get32())
        return v.view(np.float32)

    magic = fileobj.read(4)
    if magic != 'Iout':
        raise IOError('Magic number not found')
    version = get16()

    # It seems that the roi type field occupies 2 Bytes, but only one is used
    roi_type = get8()
    # Discard second Byte:
    get8()

    if not (0 <= roi_type < 11):
        raise ValueError('roireader: ROI type %s not supported' % roi_type)

    if roi_type != 7:
        raise ValueError('roireader: ROI type %s not supported (!= 7)' % roi_type)

    top = get16()
    left = get16()
    bottom = get16()
    right = get16()
    n_coordinates = get16()

    x1 = getfloat() 
    y1 = getfloat() 
    x2 = getfloat() 
    y2 = getfloat()
    stroke_width = get16()
    shape_roi_size = get32()
    stroke_color = get32()
    fill_color = get32()
    subtype = get16()
    if subtype != 0:
        raise ValueError('roireader: ROI subtype %s not supported (!= 0)' % subtype)
    options = get16()
    arrow_style = get8()
    arrow_head_size = get8()
    rect_arc_size = get16()
    position = get32()
    header2offset = get32()

    if options & SUB_PIXEL_RESOLUTION:
        getc = getfloat
        points = np.empty((n_coordinates, 2), dtype=np.float32)
    else:
        getc = get16
        points = np.empty((n_coordinates, 2), dtype=np.int16)
    points[:,1] = [getc() for i in xrange(n_coordinates)]
    points[:,0] = [getc() for i in xrange(n_coordinates)]
    points[:,1] += left
    points[:,0] += top
    points -= 1
    return points

def read_roi_zip(fname):
    import zipfile
    with zipfile.ZipFile(fname) as zf:
        return [read_roi(zf.open(n))
                    for n in zf.namelist()]