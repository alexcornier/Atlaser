import openslide as ops

import numpy as np

from skimage.filters import threshold_otsu

from skimage.morphology import binary_closing, erosion, dilation

from skimage.measure import label, regionprops

from skimage.util import img_as_uint

from skimage.color import rgb2gray

from PIL import Image

from pathlib import Path

import operator





def get_bboxes(im):

    """

    Detect brain slices on a DAPI channel in an open ndpi file



    Parameters

    ----------

    im: Openslide Image

        Open ndpi image



    Returns

    -------

    bboxes: 2D Numpy array

        Each row is a bounding box: top-left corner, width and height to be given to OpenSlide.read_region

    """

    dims = im.dimensions[::-1]   #Créer un tuple (height, width) de l'image im

    r = 100

    pic = np.asarray(im.get_thumbnail((dims[0], dims[1] / r)))  #Créer un thumbnail de im de dimension height/width/r et le transforme en array

    ratio = np.mean([d / s for d, s in zip(dims, pic.shape)])  #Calcul le ration moyen de l'image

    th = threshold_otsu(pic[..., 2])  #Return un float, tous les pixel de pic qui sont au dessus de th sont au 1er plan

    binary = dilation(erosion(binary_closing(pic.max(2) > th, np.ones((11, 11))), np.ones((5, 5))), np.ones((11, 11)))  
    #binary est un tuple de 0 et 1 qui décrit l'image après avoir sortir les defauts (dilatation erosion closing)

    l_im = label(binary)  #return un ndarray avec un label pour chaque pixel de binary qui sont connecté entre eux

    obj = [o for o in regionprops(l_im) if o.area > 5000]

    bboxes = np.array([o.bbox for o in obj]) * ratio

    bboxes = np.intp(np.apply_along_axis(lambda x: [x[1], x[0], (x[3] - x[1]), (x[2] - x[0])], 1, bboxes))



    return bboxes





def crop(im, bb, res):

    """

    Crop on brain slice from an ndpi image, given the bounding box around it



    Parameters

    ----------

    im: openslide.OpenSlide

        Open ndpi file

    bb: tuple

        Bounding box

    res: int

        Resolution level wanted



    Returns

    -------

    crop_im: PIL.Image

        Cropped image

    """



    crop_region = im.read_region(bb[:2], res,

                                 np.intp(bb[2:] / im.level_downsamples[res]))  # (x,y top-left), resolution, (width, height)

    crop_region = rgb2gray(np.asarray(crop_region))

    crop_im = Image.fromarray(crop_region)

    return crop_im




def crop_all(im, bboxes, res, path):

    """

    Crop an open ndpi image based on given computed bounding boxes. Save the cropped images in tiff files



    Parameters

    ----------

    im: openslide.OpenSlide

        Open ndpi file

    bboxes: numpy array

        Array of bounding boxes as returned by get_bboxes

    res: int

        Level of resolution to select from

    path: PosixPath

        Path of the original ndpi file



    """

    # Make sure the resolution level stays within bounds

    res = np.clip(res, 0, im.level_count-1)

    # Resize the bboxes according to the resolution level



    for ix, bb in enumerate(bboxes):

        # logger.info(f'\t Slice #{ix}')

        crop_im = crop(im, bb, res)

        crop_im.save(path.parent / f'{path.stem}_{ix}.tiff')





def parse_ndpis(filepath):

    """

    Parse an ndpis file which is just a simple text file listing all related ndpi images



    Parameters

    ----------

    filepath: str

        Path to the ndpis file



    Returns

    -------

    prms: dict

        All the parameters stored in the ndpis file

        In the file they are organized as key=value, we split that into a dict

    """

    filepath = Path(filepath)

    if filepath.suffix != '.ndpis':

        # logger.error(f'File passed to parse is not ndpis. It is {filepath}')

        raise ValueError('File not supported. Has to be ndpis')

    prms = {'path': filepath.parent}

    with open(filepath, 'r') as f:

        for line in f:

            line = line.strip()

            if line.startswith('['):

                continue

            sl = line.split('=')

            prms[sl[0]] = ''.join(sl[1:])

    return prms





def open_im(im_path):

    im = ops.OpenSlide(im_path.as_posix())



    return im





def crop_from_dapi(prms, res):

    """

    Crop all the related files, listed in the parameter dictionary recovered from an ndpis file

    based on the dapi channel (dapi has to be in the name of one image)

    Detect brain slices, and crop all channels with the same ROIs

    All the cropped images are saved as tiff files



    Parameters

    ----------

    prms:: dict

        As returned from parse_ndpis

    res: int

        Resolution level to crop



    """

    images = [prms[f'Image{ix}'] for ix in range(int(prms['NoImages']))]

    dapi_im = [im for im in images if 'dapi' in im.lower()][0]

    im_path = prms['path'] / dapi_im

    im = ops.OpenSlide(im_path.as_posix())

    bboxes = get_bboxes(im)

    im.close()

    for im in images:

        # logger.info(f'Cropping: {im}')

        im_path = prms['path'] / im

        im = ops.OpenSlide(im_path.as_posix())

        crop_all(im, bboxes, res, im_path)

        im.close()

# #Martin
# def resize_ndpi(im_path):

#     im = ops.OpenSlide(im_path.as_posix())

#     dimlvl = im.level_dimensions

#     im_downscale = im.read_region((0, 0), 3 , dimlvl[3])

#     im_downscale.save(im_path.parent / f'{im_path.stem}.tiff')


    

