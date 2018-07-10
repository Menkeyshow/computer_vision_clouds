import os
import skimage
#own modules
import box_clouds as box

def ensure_directory_exists(dirname):
    """Checks if the directory `dirname` exists, and creates it if it
    doesn't.

    Returns `True` if the directory existed already, and `False` if it
    had to be created."""

    if os.path.exists(dirname):
        return True
    else:
        os.makedirs(dirname)
        return False


cloud_kinds = {
    "stratiform":
        ["cirrostratus", "altostratus", "nimbostratus", "stratus"],
    "cirriform": ["cirrus"],
    "stratocumuliform": ["cirrocumulus", "altocumulus", "stratocumulus"],
    "cumuliform": ["cumulus", "cumulonimbus"]
}

def cropImageArray(array):
    '''
    Returns a cropped imageArray, cropped with the box_clouds method.
    '''
    cropped_array = []
    for img in array:
        image_shape = box.binarized_crop(img, 0.2)
        #print("image_shape:" ,image_shape)
        #print("img_shape:" , img.shape)
        image = img[0:image_shape[0],0:image_shape[1],:]
        #print("image:",image.shape)
        resized = skimage.transform.resize(image, (500,500))
        #print("resized:",resized.shape)
        cropped_array.append(resized)
    return cropped_array