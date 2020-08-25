import numpy as np
import nrrd
from tqdm import trange, tqdm
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter
import json
import matplotlib.pyplot as plt
#import SimpleITK as sitk


def get_atlas(path):
    d, h = nrrd.read(path)
    return d


def contourify(stack):
    """
    Get the  contours of the regions in the atlas
    Very slow. Should be done once and the result saved

    Parameters
    ----------
    stack: numpy ndarray
        Atlas annotations

    Returns
    -------
    contoured: numpy ndarray
        Contoured atlas
    """

    contoured = np.zeros(stack.shape, dtype=stack.dtype)
    for ix in trange(stack.shape[2]):
        c_slice = stack[:, :, ix]
        structures_ix = np.unique(c_slice.reshape(-1))
        for c_struct in tqdm(structures_ix):
            if c_struct == 0:
                continue
            contours = find_contours(c_slice, c_struct - .5)
            for cnt in contours:
                rr, cc = polygon_perimeter(*cnt.T, c_slice.shape)
                contoured[rr, cc, ix] = c_struct
    return contoured


def read_ontology(path):
    with open(path, 'r') as f:
        an = json.load(f)
    return an['msg']


def id_colors(onto, colors={}):
    """
    From an ontology return the colors corresponding to a structure

    Parameters
    ----------
    onto: list
        Ontology as returned from ::py:func: `read_ontology`
    colors: dict
        Used because function is recursive. Should be left at default when called from outside

    Returns
    -------
    colors: dict
    """
    for s in onto:
        colors[s['id']] = s['color_hex_triplet']
        id_colors(s['children'])
    return colors


def color_atlas(atlas, colors):
    ids = np.unique(atlas)
    c_atlas = np.zeros(atlas.shape + (3,), dtype=np.uint8)
    for ix in tqdm(ids):
        if ix == 0:
            continue
        try:
            color = int(colors[ix], 16)
            r = color >> 16
            g = (color - (r << 16)) >> 8
            b = color - (r << 16) - (g << 8)
        except KeyError:
            r, g, b = 255, 255, 255
            print(ix)
        gi = atlas == ix
        c_atlas[gi, 0] = r
        c_atlas[gi, 1] = g
        c_atlas[gi, 2] = b

    return c_atlas


def save_color_atlas():
    onto = read_ontology("mouse_ontology.json")
    colors = id_colors(onto)
    atlas = np.load('contourify.npy')
    c_atlas = color_atlas(atlas, colors)
    np.save('color_atlas.npy', c_atlas)


# def get_registration_method():
#     registration_method = sitk.ImageRegistrationMethod()
#     # Similarity metric settings.
#     registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
#     registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
#     registration_method.SetMetricSamplingPercentage(.05)
#
#     registration_method.SetInterpolator(sitk.sitkLinear)
#     # Optimizer settings.
#     # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=0.10, numberOfIterations=100)
#     registration_method.SetOptimizerAsGradientDescent(learningRate=0.10, numberOfIterations=100,
#                                                       convergenceMinimumValue=1e-12, convergenceWindowSize=10)
#     # registration_method.SetOptimizerScalesFromIndexShift(smallParameterVariation=1e-4)
#     registration_method.SetOptimizerScalesFromJacobian()
#
#     # Setup for the multi-resolution framework.
#     registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
#     registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
#     # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
#      # DEBUG
#     registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
#     registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
#     registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
#     registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))
#
#     return registration_method
#
#
# def get_img_center(image):
#     dv, ml = np.where(image > image.mean())
#     return int(dv.min()), int(ml.mean())
#
#
# def get_slice_border(image):
#     dv, ml = np.where(image > image.mean())
#     return int(dv.min()), int(ml.min ())
#
#
# def register(template_volume, image, transf):
#     import matplotlib.pyplot as plt
#     plt.ion()
#     n_bef = template_volume.shape[2] // 2
#     # Including the picture to align in a 3D stack of the same size as the template volume
#     im_vol = np.zeros(image.shape + (template_volume.shape[2],), dtype=image.dtype)
#     im_vol[:, :, n_bef + 1] = image
#
#     # f_img = sitk.GetImageFromArray(template_volume.transpose(2, 1, 0))
#     # m_img = sitk.GetImageFromArray(im_vol.transpose((2, 1, 0)))
#     f_img = sitk.GetImageFromArray(template_volume[:, :, n_bef].T)
#     m_img = sitk.GetImageFromArray(image.T)
#     f_img = sitk.Cast(f_img, sitk.sitkFloat32)
#     m_img = sitk.Cast(m_img, sitk.sitkFloat32)
#     # f_img.SetOrigin(get_img_center(template_volume[:, :, n_bef]))
#     f_img.SetOrigin((0, 0))
#     m_img.SetOrigin((525-67, 1150-121))
#     f_img.SetSpacing((25, 25))
#     px_size = 0.454
#     m_img.SetSpacing((px_size, px_size))
#     print(f_img.GetOrigin(), m_img.GetOrigin())
#     rif = sitk.ResampleImageFilter()
#     rif.SetDefaultPixelValue(-100)
#     rif.SetReferenceImage(f_img)
#     rif.SetOutputSpacing(f_img.GetSpacing())
#     rif.SetInterpolator(sitk.sitkLinear)
#     print('Resampling')
#     d_img = rif.Execute(m_img)
#     print(d_img.GetOrigin())
#     plt.figure()
#     plt.imshow(sitk.GetArrayFromImage(m_img).T)
#     plt.figure()
#     plt.imshow(sitk.GetArrayFromImage(d_img).T)
#     plt.figure()
#     plt.imshow(sitk.GetArrayFromImage(f_img).T)
#     # return image
#     # # ******
#     # # Transformations defined in GUI
#     # scaling = affine_scale(1/transf.scale, 1/transf.scale)
#     # translation = affine_translate(transf.translation[1], transf.translation[0])
#     # rotation = affine_rotate(m_img, transf.rotation)
#
#     # # Complete composite transformation
#     # composite_transform = sitk.Transform(scaling)
#     # composite_transform.AddTransform(rotation)
#     # composite_transform.AddTransform(translation)
#     # r_img = resample(m_img, composite_transform)
#     # # Cropping the resulting image
#     # crop_filter = sitk.CropImageFilter()
#     # a_width, a_height = f_img.GetWidth() // 2, f_img.GetHeight() // 2
#     # bounds = np.array([[-a_width, -a_height, 0], [a_width, a_height, r_img.GetDepth()]])
#     # bounds -= np.tile(np.array(r_img.GetOrigin(), dtype=np.int64), (2, 1))
#     # bounds = [[int(x) for x in b] for b in bounds]
#     # r_img = crop_filter.Execute(r_img, bounds[0], bounds[1])
#     # result_image = sitk.GetArrayFromImage(r_img)
#     # # plt.imshow(result_image.max(0).T)
#     # # return result_image.max(0)
#     #
#     # ********
#
#     # Registration
#     registration_method = get_registration_method()
#     initial_transform = sitk.CenteredTransformInitializer(f_img,
#                                                           d_img,
#                                                           sitk.Euler2DTransform(),
#                                                           sitk.CenteredTransformInitializerFilter.MOMENTS)
#     registration_method.SetMovingInitialTransform(initial_transform)
#     registration_method.SetInitialTransform(sitk.Similarity2DTransform())
#     try:
#         final_transform = registration_method.Execute(sitk.Cast(f_img, sitk.sitkFloat32),
#                                                       sitk.Cast(d_img, sitk.sitkFloat32))
#     except:
#         print('error')
#         final_transform = sitk.Transform()
#     final_transform.AddTransform(initial_transform)
#     moving_resampled = sitk.Resample(m_img, f_img, final_transform, sitk.sitkCosineWindowedSinc, 0.0,
#                                      m_img.GetPixelID())
#
#     print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
#     print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
#     print(f'Transformation: {final_transform.GetParameters()}')
#
#     moving_resampled = sitk.Resample(moving_resampled, f_img, final_transform, sitk.sitkCosineWindowedSinc, 0.0, m_img.GetPixelID())
#     result_image = sitk.GetArrayFromImage(moving_resampled)
#     print(result_image.shape, template_volume.shape)
#     plt.figure()
#     plt.imshow(result_image.T)
#
#     return result_image
#
#     param_map = sitk.VectorOfParameterMap()
#     param_map.append(sitk.GetDefaultParameterMap("affine"))
#     # param_map.append(sitk.GetDefaultParameterMap("rigid"))
#     # param_map.append(sitk.GetDefaultParameterMap("bspline"))
#     elastix_filter = sitk.ElastixImageFilter()
#     elastix_filter.LogToFileOn()
#     elastix_filter.SetFixedImage(f_img)
#     elastix_filter.SetMovingImage(moving_resampled)
#     elastix_filter.SetParameterMap(param_map)
#     elastix_filter.Execute()
#     result_image = sitk.GetArrayFromImage(elastix_filter.GetResultImage()).T
#     # result_image = result_image.max(2).T
#     # plt.imshow((result_image.T + template_volume[:, :, 3]) * .5)
#     # print(result_image.shape)
#
#     return result_image
#
#
# # Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
# # metric_values list.
# def update_multires_iterations():
#     global metric_values, multires_iterations
#     multires_iterations.append(len(metric_values))
#
#
# # Callback invoked when the StartEvent happens, sets up our new data.
# def start_plot():
#     global metric_values, multires_iterations
#     plt.figure()
#     metric_values = []
#     multires_iterations = []
#
#
# # Callback invoked when the EndEvent happens, do cleanup of data and figure.
# def end_plot():
#     global metric_values, multires_iterations
#
#     del metric_values
#     del multires_iterations
#
#
# # Callback invoked when the IterationEvent happens, update our data and display new figure.
# def plot_values(registration_method):
#     global metric_values, multires_iterations
#
#     metric_values.append(registration_method.GetMetricValue())
#     # Clear the output area (wait=True, to reduce flickering), and plot current data
#     # Plot the similarity metric values
#     plt.plot(metric_values, 'r')
#     plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
#     plt.xlabel('Iteration Number', fontsize=12)
#     plt.ylabel('Metric Value', fontsize=12)
#     plt.show()
#
#
# def resample(image, transform):
#     # Output image Origin, Spacing, Size, Direction are taken from the reference
#     # image in this call to Resample
#     side = max(image.GetWidth(), image.GetHeight())
#     reference_image = sitk.Image(side, side, image.GetDepth(),
#                                  image.GetPixelID())
#     reference_image.SetOrigin((-reference_image.GetWidth()//2, -reference_image.GetHeight()//2, 0))
#     # interpolator = sitk.sitkCosineWindowedSinc
#     interpolator = sitk.sitkLinear
#     default_value = 100.0
#     r_sampled = sitk.Resample(image, reference_image, transform,
#                               interpolator, default_value)
#     return r_sampled
#
#
# def affine_rotate_2D(image, degrees=15.0):
#     transform = sitk.AffineTransform(2)
#     parameters = np.array(transform.GetParameters())
#     new_transform = sitk.AffineTransform(transform)
#     matrix = np.array(transform.GetMatrix()).reshape((2, 2))
#     radians = np.pi * degrees / 180.
#     rotation = np.array([[np.cos(radians), -np.sin(radians)],
#                          [np.sin(radians), np.cos(radians)]])
#     new_matrix = np.dot(rotation, matrix)
#     new_transform.SetMatrix(new_matrix.ravel())
#     new_transform.SetCenter((image.GetWidth() / 2, image.GetHeight() / 2))
#     resampled = resample(image, new_transform)

#     return new_transform
#
#
# def affine_rotate(image, degrees=15.0):
#     transform = sitk.AffineTransform(3)
#     parameters = np.array(transform.GetParameters())
#     new_transform = sitk.AffineTransform(transform)
#     matrix = np.array(transform.GetMatrix()).reshape((3, 3))
#     radians = np.pi * degrees / 180.
#     rotation = np.array([[np.cos(radians), -np.sin(radians), 0],
#                          [np.sin(radians), np.cos(radians), 0],
#                          [0, 0, 1]])
#     new_matrix = np.dot(rotation, matrix)
#     new_transform.SetMatrix(new_matrix.ravel())
#     new_transform.SetCenter((image.GetWidth()/2, image.GetHeight()/2, 0))
#     # resampled = resample(image, new_transform)
#
#     return new_transform
#
#
# def affine_translate(x_translation=3.1, y_translation=4.6):
#     transform = sitk.AffineTransform(3)
#     new_transform = sitk.AffineTransform(transform)
#     new_transform.SetTranslation((x_translation, y_translation, 0))
#     return new_transform
#
#
# def affine_translate_2D(x_translation=3.1, y_translation=4.6):
#     transform = sitk.AffineTransform(2)
#     new_transform = sitk.AffineTransform(transform)
#     new_transform.SetTranslation((x_translation, y_translation))
#     return new_transform
#
#
# def affine_scale(x_scale=3.0, y_scale=0.7):
#     transform = sitk.AffineTransform(3)
#     new_transform = sitk.AffineTransform(transform)
#     matrix = np.array(transform.GetMatrix()).reshape((3, 3))
#     matrix[0, 0] = x_scale
#     matrix[1, 1] = y_scale
#     matrix[2, 2] = 1
#     new_transform.SetMatrix(matrix.ravel())
#     return new_transform
#
#
# def affine_scale_2D(x_scale=3.0, y_scale=0.7):
#     transform = sitk.AffineTransform(2)
#     new_transform = sitk.AffineTransform(transform)
#     matrix = np.array(transform.GetMatrix()).reshape((2, 2))
#     matrix[0, 0] = x_scale
#     matrix[1, 1] = y_scale
#     new_transform.SetMatrix(matrix.ravel())
#     return new_transform
#
#
# def normalize(x):
#     """
#     Normalize an array between 0 and 1
#
#     Parameters
#     ----------
#     x: Numpy ndarray
#
#     Returns
#     -------
#     n_x: Numpy ndarray
#         Normalized array
#     """
#
#     mv = x.min()
#     n_x = (x - mv) /(x.max()-mv)
#     return n_x
#
#
# if __name__ == '__main__':
#     from PIL import Image
#     import json
#     from controls import Transform
#     Image.MAX_IMAGE_PIXELS = None
#     im = Image.open('coronal.tiff')
#     dpath = 'params_coronal.json'
#     with open(dpath, 'r') as f:
#         d_transf = json.load(f)['manual']
#         transf = Transform((d_transf['x_shift'], d_transf['y_shift']), d_transf['rotation'],
#                            d_transf['scale'])
#     dw_img = im.resize((im.width//5, im.height//5))
#     image = np.array(im)
#     image = normalize(np.float32(image))
#     a = get_atlas('average_template_25.nrrd')
#     c_pos = 206
#     template_volume = a[c_pos-2:c_pos+3, : , :].transpose((1,2, 0))
#     template_volume = normalize(np.float32(template_volume))
#     n_bef = template_volume.shape[2] // 2
#     f_img = sitk.GetImageFromArray(template_volume[:, :, n_bef].T)
#     m_img = sitk.GetImageFromArray(image.T)
#     f_img = sitk.Cast(f_img, sitk.sitkFloat32)
#     m_img = sitk.Cast(m_img, sitk.sitkFloat32)
#     # f_img.SetOrigin(get_img_center(template_volume[:, :, n_bef]))
#     f_img.SetOrigin((0, 0))

