# choose a random image from training and color normalize the dataset
import numpy as np
from PIL import Image


# Arrays for color conversion. Note that LMS is an intermediate color space between RGB and LAB.
rgb2lms = np.array([[0.3811, 0.5783, 0.0402],
                    [0.1967, 0.7244, 0.0782],
                    [0.0241, 0.1288, 0.8444]])

lms2lab = np.matmul(
        np.array([[1/np.sqrt(3), 0, 0],
                  [0, 1/np.sqrt(6), 0],
                  [0, 0, 1/np.sqrt(2)]]),
        np.array([[1, 1, 1],
                  [1, 1, -2],
                  [1, -1, 0]]))

lms2rgb = np.array([[4.4679, -3.5873, 0.1193],
                    [-1.2186, 2.3809, -0.1624],
                    [0.0497, -0.2439, 1.2045]])
    
lab2lms = np.matmul(
        np.array([[1, 1, 1],
                  [1, 1, -1],
                  [1, -2, 0]]),
        np.array([[np.sqrt(3)/3, 0, 0],
                  [0, np.sqrt(6)/6, 0],
                  [0, 0, np.sqrt(2)/2]]))


def rgb_to_lab(im):
    """rgb_to_lab()
    Converts an RGB image into the LAB color space.
    
    INPUTS
    ------
    im : array-like
        RGB image in numpy form, with values from 0 to 255 range.
        
    RETURNS
    -------
    lab_im : array-like
        Image in LAB color space.
    """    
    # the conversion happens at the pixel level, reshape to 2D matrix
    n_row, n_col = im.shape[0:2]
    n_px = n_row * n_col
    im = np.reshape(im.copy(), (n_px, 3))
    # convert to LMS cone space
    lms_im = np.matmul(rgb2lms, im.T)
    # convert to LAB color space, remember LMS values need to be logged
    lab_im = np.matmul(lms2lab, np.log(lms_im)).T
    lab_im = np.reshape(lab_im, (n_row, n_col, 3)) # reshape back to 3D matrix
    return lab_im


def lab_color_stats(im):
    """lab_color_stats()
    Calculates the LAB color space mean and standard deviation for each channels
    form a given image.
    
    INPUTS
    ------
    im : array-like or str
        You can either give the path to the image or the image in numpy array form.
        This will be the image which the LAB color stats are calculated from. Note that
        the array should be in RGB with values from 0 to 255.
        
    RETURNS
    -------
    means : 1D-array
        Mean of each LAB channel.
    stds : 1D-array
        Standard deviation of each LAB channel.
    """
    # Check if im is a path.
    if type(im) is str:
        # Read the image into a numpy form.
        image = np.array(Image.open(im))
    else: 
        image = im.copy() # make a copy of the image, avoids potential referencing
    
    # Convert image to LAB color space.
    lab_im = rgb_to_lab(image)
    # Calculate the mean and standard deviation of each channel
    means = np.mean(lab_im, axis=(0,1))
    stds = np.std(lab_im, axis=(0,1))
    
    return means, stds


def reinhard_normalize(im_src, target_mu, target_sigma, src_mu=None, src_sigma=None):
    """reinhard_normalize()
    THIS FUNCTION NEEDS LOOKING OVER TO MAKE IT 100% IS WORKING WELL
    ... coming soon ...
    """
    
    m = im_src.shape[0]
    n = im_src.shape[1]
    
    # convert input image to LAB color space
    im_lab = rgb_to_lab(im_src)
    
    # calculate src_mu if not provided
    if src_mu is None:
        src_mu = im_lab.sum(axis=0).sum(axis=0) / (m * n)

    # center to zero-mean
    for i in range(3):
        im_lab[:, :, i] = im_lab[:, :, i] - src_mu[i]

    # calculate src_sigma if not provided
    if src_sigma is None:
        src_sigma = ((im_lab * im_lab).sum(axis=0).sum(axis=0) /
                     (m * n - 1)) ** 0.5

    # scale to unit variance
    for i in range(3):
        im_lab[:, :, i] = im_lab[:, :, i] / src_sigma[i]

    # rescale and recenter to match target statistics
    for i in range(3):
        im_lab[:, :, i] = im_lab[:, :, i] * target_sigma[i] + target_mu[i]

    # convert back to RGB colorspace
    im_normalized = lab_to_rgb(im_lab)
    im_normalized[im_normalized > 255] = 255
    im_normalized[im_normalized < 0] = 0
    im_normalized = im_normalized.astype(np.uint8)
    
    return im_normalized 


def lab_to_rgb(lab_im):
    """lab_to_rgb()
    THIS FUNCTION NEEDS LOOKING OVER TO MAKE IT 100% IS WORKING WELL
    Converts an RGB image into the LAB color space, as outlined in 
    Reinhard et al 2001.
    
    INPUTS
    ------
    lab_im : array-like
        3D numpy array representing a LAB image.
        
    RETURNS
    -------
    lab_im : array-like
        Converted image to RGB color space.
    """    
    # the conversion happens at the pixel level, reshape to 2D matrix
    n_row, n_col = lab_im.shape[0:2]
    n_px = n_row * n_col
    lab_im = np.reshape(lab_im, (n_px, 3))
    # convert to LMS cone space
    lms_im = np.matmul(lab2lms, lab_im.T)
    # convert to RGB color space, remember LMS values need to be logged
    rgb_im = np.matmul(lms2rgb, np.exp(lms_im)).T
    rgb_im = np.reshape(rgb_im, (n_row, n_col, 3)) # reshape back to 3D matrix
    return rgb_im
