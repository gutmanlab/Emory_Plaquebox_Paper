"""Functions for analyzing confidence heatmaps in a pathological relevant fashion"""
import cv2
from skimage import measure
from skimage.transform import resize
import numpy as np
import imageio
import pandas as pd


def conf_heatmap_fov_scores(blob_mask_path, save_path, fov_shape=(128, 128), stride=16, 
                            labels=('cored', 'diffuse', 'caa')):
    """Process a blob mask of an image (obtained from blob counting on CNN confidence heatmaps) by analyzing it
    in small field of view (FOV) regions. The fov shape is slid across the blob mask and number of unique blobs
    is reported for that FOV. At the edges the stride may be smaller to fit all regions of the image and avoid 
    missing plaques. The output is saved as a csv file with blob count information for all FOV with at least one
    plaque for all three blob masks (cored, diffuse, and caa types).

    Columns - [x, y, fov height, fov width, type of plaque, blob count] where x and y are the top left corner 
    of the FOV
    
    :param blob_mask_path : str
        path to blob mask
    :param save_path : str
        path to save csv output to
    :param fov_shape : tuple (default of 128 by 128)
        size of FOV to use
    :param stride : int (default of 16)
        stride to use
    :param labels : tuple (default of cored, diffuse, and caa)
        labels to use for csv output
    """
    count_data = []

    # load blob mask
    blob_masks = np.load(blob_mask_path)
    height, width = blob_masks.shape[1:]

    for i in range(3):    
        """Blob mask is a numpy array with each individual blob containing unique pixel values. To obtain a count
        of unique blobs all you have to do is count he unique values in the array - minus 0 for background."""
        blob_mask = blob_masks[i]

        # slide across blob mask given FOV shape and stride
        for x in range(0, width, stride):
            # shift left if at edge
            if x + fov_shape[1] > width:
                x = width - fov_shape[1]

            for y in range(0, height, stride):
                # shift up if at edge
                if y + fov_shape[0] > height:
                    y = height - fov_shape[0]

                # select region of blob mask for this FOV
                fov_im = blob_mask[y:y+fov_shape[0], x:x+fov_shape[1]]

                # get a count of unique blobs
                unique_values = np.unique(fov_im)
                blob_count = len(unique_values)

                # remove 1 count if 0 label is in the FOV
                if np.isin(0, unique_values):
                    blob_count -= 1

                # add these count to list if there is at least one blob
                if blob_count > 0:
                    count_data.append([x, y, fov_shape[0], fov_shape[1], labels[i], blob_count])
                    
    # save output to csv
    df = pd.DataFrame(count_data, columns=['x', 'y', 'height', 'width', 'label', 'count'])
    df.to_csv(save_path, index=False)


def label_conf_heatmap(conf_heatmap, conf_threshold, size_threshold=None):
    """Provided a confidence heatmap of an image, threshold it to a binary mask and label pixels in proximity
    of each other by unique labels. This provides a means to group potential objects together by giving all 
    pixels of that object a unique label.
    
    :param conf_heatmap : numpy array
        confidence heatmap (a float dtype 2D array) of values from 0 to 1
    :param conf_threshold : float
        value to use to thershold the confidence heatmap to a binary mask
    :param size_threshold : int (default: None)
        value used to threshold small objects after label assignment
        
    :return labeled_mask : numpy array
        dtype int64 labeled mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    
    # threshold to binary mask
    binary_mask = (conf_heatmap > conf_threshold).astype(np.float32)
    
    # morphological binary operations (noise removal) [values are 0. and 1.]
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # label cluster of pixels uniquely
    labeled_mask = measure.label(binary_mask, neighbors=8, background=0)
    
    if size_threshold is not None:
        # remove group of pixels that are not large enough
        for label in np.unique(labeled_mask):
            if label == 0:
                continue
            
            # zero out label pixels if less than threshold
            label_pixels = labeled_mask == label
            
            if np.count_nonzero(label_pixels) < size_threshold:
                labeled_mask[label_pixels] = 0           

    return labeled_mask
            

def cerad_style_cnn_scores(labeled_mask, area_mask, kernel):
    """From a labeled mask (see label_conf_heatmap(..)) calculate CNN scores in a kernel style approach. Kernel
    is a rectangular window that slides accross the labeled mask, no overlap, and a CNN score is calulated 
    for each kernel. The area mask denotes the region of interest, all regions outside this mask are ignored
    in the calculation of the scores.
    
    :param labeled_mask : numpy array
        dtype int64 labeled mask
    :param area_mask : area_mask
        binary mask denoting the region of interest
    :param kernel : tuple
        (width, height) of the kernel.
        
    :return locs : list
        the (x, y) top left corner of each kernel
    :return scores : list
        the corresponding CNN scores for each kernel in locs
    :return cerad_score_map : numpy array
        the heatmap of the CNN scores, as calculated by the kernel approach
    """
    labeled_mask = labeled_mask.copy()
    area_mask = area_mask.copy()
    
    # check that mask are correct size, otherwise reshape the area mask
    if labeled_mask.shape != area_mask.shape:
        area_mask = resize(area_mask, labeled_mask.shape, mode='constant')
        
    # for devel turn this skip variable to True to avoid edges where kernel would not be the right size
    skip = False
        
    locs = []
    scores = []
    kernel_threshold = (kernel[0] * kernel[1]) / 2
    
    # create the CERAD-style heatmap of scores
    cerad_score_map = np.zeros(labeled_mask.shape, dtype=np.float64)
    
    # slide the kernel (no overlap) accross the image and avoid the edges if not enough space for kernel
    for x in range(0, labeled_mask.shape[1], kernel[0]):
        # skip if at edge
        if skip and x + kernel[0] >= labeled_mask.shape[1]:
            continue
        for y in range(0, labeled_mask.shape[0], kernel[1]):
            if skip and y + kernel[1] >= labeled_mask.shape[0]:
                continue
            
            # apply the binary mask
            l = labeled_mask[y: y + kernel[1], x: x + kernel[0]]
            m = area_mask[y: y + kernel[1], x: x + kernel[0]]
            
            m_count = np.count_nonzero(m)
            
            # at least 50% of kernel area must be in mask
            if m_count < kernel_threshold:
                continue
            
            # apply area mask to label mask
            l = cv2.bitwise_and(l, l, mask=m)
            
            # count number of unique labels -> count of pathology in the kernel
            labels = list(np.unique(l))
            count = len(labels)
            if 0 in labels:
                count -= 1
                
            # normalize by number of gray matter pixels (region of interest) in the kernel 
            score = count * 1000 / m_count
            cerad_score_map[y: y + kernel[1], x: x + kernel[0]] = score
            
            scores.append(score)
            locs.append((x,y))
                
    # sort locations and scores in decreasing order
    scores, locs = (list(t) for t in zip(*sorted(zip(scores, locs))))
    scores = list(reversed(scores))
    locs = list(reversed(locs))
    return locs, scores, cerad_score_map


def count_blobs(heatmap_path, confidence_thresholds, pixel_thresholds, mask_path=None):
    """From confidence heatmaps, count the blobs in the image. Confidence heatmaps are thresholded to binary
    masks to remove noise and cleaned up with binary operations. Pixels are grouped together by distance metric
    into unique blobs and the number of unique blobs is counted and returned with labeled mask. Blobs that are
    below a pixel threshold in size are removed. Optinal: give a path to a binary region of interest (ROI) mask to 
    also return the count of blobs that fall within this ROI. Note that a blob is considered to fall inside the
    ROI if at least one pixel is inside the ROI.
    
    :param heatmap_path : str
        path to confidence heatmaps
    :param confidence_thresholds : list
        length of 3 for confidence thresholds for each heatmap
    :param pixel_thresholds : list
        length of 3 for blob size minimum threshold for each heatmap
    :param mask_path : str (default: None)
        mask to binary image for ROI thresholding
        
    :return blob_masks : ndarray
        the cleaned up blob masks for entire image
    :return blob_counts : int
        number of blobs in blob_masks
    :return roi_blob_masks : ndarray
        blob_masks after zeroing values outside of provided ROI mask. None is mask_path not provided
    :return roi_blob_counts : int
        number of blobs in roi_blob_masks. None is mask_path not provided
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    heatmaps = np.load(heatmap_path)
    
    if mask_path is not None:
        roi_mask = imageio.imread(mask_path)
    else:
        roi_mask = None
    
    # loop through each heatmap
    blob_masks = np.zeros(heatmaps.shape, dtype=np.uint32)
    roi_blob_masks = np.zeros(heatmaps.shape, dtype=np.uint32)
    blob_counts = [0]*3
    roi_blob_counts = [0]*3
    
    for i, confidence_threshold, pixel_threshold in zip(range(3), confidence_thresholds, pixel_thresholds):
        # threshold and clean mask
        mask = (heatmaps[i] > confidence_threshold).astype(np.float32)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # label pixels by unique values for each group of pixels using a distance metric criteria
        # these grouped pixels (of same label) are referred to as blobs
        label_mask = measure.label(mask, neighbors=8, background=0)
        
        # remove small blobs
        for label in np.unique(label_mask):
            if label == 0:  # background
                continue
                
            # get the indices with this label and the count
            label_indices = (label_mask == label)
            label_count = np.count_nonzero(label_indices)
            
            # include if bigger than threshold
            if label_count > pixel_threshold:
                blob_counts[i] += 1
                blob_masks[i, label_indices] = label
        
        # if roi mask then provide a roi_sizes
        if roi_mask is not None:
            # using roi mask, zero any pixels outside of roi mask in the standard blob mask
            # this new filtered mask can be used to calculate unique count of blobs
            temp = blob_masks[i].copy()
            temp[roi_mask == 0] = 0
            roi_blob_masks[i] = temp
            roi_blob_counts[i] = len(np.unique(roi_blob_masks[i])) - 1
    
    if roi_mask is None:
        roi_blob_masks = None
        roi_blob_counts = None
        
    return blob_masks, blob_counts, roi_blob_masks, roi_blob_counts