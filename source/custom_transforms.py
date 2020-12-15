import cv2
import torch
import numpy as np


def open_image_from_path(image_path):
    """
    returns image as numpy array (H, W, RGB)
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def constructMask(W, H):
    """Constructs a mask to only take into consideration a part of the frame.
    In this case it's the road. """

    mask = np.zeros(shape = (H,W), dtype = np.uint8)
    mask.fill(255)  
    cv2.rectangle(mask, (0, 0), (W, H), (0, 0, 0), -1)

    x_top_offset = 240
    x_btm_offset = 65

    poly_pts = np.array([[[640-x_top_offset, 250], [x_top_offset, 250], [x_btm_offset, 350], [640-x_btm_offset, 350]]], dtype=np.int32)
    cv2.fillPoly(mask, poly_pts, (1, 1, 1))

    return mask[130:350, 35:605]

"""
Some image augmentation methods, applied to the 2 frames before optical flow is calculated:

"""
class Crop(object):
    """
        Cropping image to test effectives of removing noise

        Variation 1:
        input: image (480 (y), 640 (x), 3) RGB
        output: image (shape is (220, 66, 3) as RGB)

        Process:
                 1) Cropping out black spots
                 3) resize to (220, 66, 3)


        Variation 2:
        input: image (480 (y), 640 (x), 3) RGB
        output: image (shape is (220, 570, 3) as RGB)
        frame_gray[130:350, 35:605]
    """
    def __call__(self, sample):
        # Crop out sky (top) (100px) and black right part (-90px)
        frame1 = sample['frame1']
        frame1 = frame1[130:350, 35:605]
        sample['frame1'] = frame1

        frame2 = sample['frame2']
        frame2 = frame2[130:350, 35:605]
        sample['frame2'] = frame2

        return sample


class GaussianBlur(object):
    """
    Apply cv2 gaussian blur for image smoothing
    """
    def __call__(self, sample):
        sample['frame1'] = cv2.GaussianBlur(sample['frame1'], (3,3), 0)
        sample['frame2'] = cv2.GaussianBlur(sample['frame2'], (3,3), 0)

        return sample

"""
Some image augmentation methods, applied to the 2 frames before optical flow is calculated:

"""
class RandomBrightness(object):
    """
    Augments the brightness of the image by multiplying the saturation by a uniform random variable
    Input: image (RGB)
    returns: image with brightness augmentation
    """
    def __init__(self):
        self.bright_factor = 1

    def __call__(self, sample):
        #augment brightness between
        bright_factor = self.bright_factor + np.clip(np.random.normal(loc=0, scale=.3), -.5, .5)
        frame1 = sample['frame1'].copy()
        frame1_hsv = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
        # perform brightness augmentation only on the second channel
        if bright_factor < 1:
            frame1_hsv[:,:,2] = frame1_hsv[:,:,2] * bright_factor
        else:
            temp_factor=2-bright_factor
            frame1_hsv[:,:,2]=frame1_hsv[:,:,2]*temp_factor+(1-temp_factor)*255
        # change back to RGB
        frame1 = cv2.cvtColor(frame1_hsv, cv2.COLOR_HSV2RGB)
        sample['frame1'] = frame1

        #switch the brightness of frame 2 by a smaller amount around frame1
        bright_factor = bright_factor + np.clip(np.random.normal(loc=0, scale=.05), -.15, .15)
        frame2 = sample['frame2'].copy()
        frame2_hsv = cv2.cvtColor(frame2, cv2.COLOR_RGB2HSV)
        # perform brightness augmentation only on the second channel
        if bright_factor < 1:
            frame2_hsv[:,:,2] = frame2_hsv[:,:,2] * bright_factor
        else:
            temp_factor=2-bright_factor
            frame2_hsv[:,:,2]=frame2_hsv[:,:,2]*temp_factor+(1-temp_factor)*255
        # change back to RGB
        frame2 = cv2.cvtColor(frame2_hsv, cv2.COLOR_HSV2RGB)
        sample['frame2'] = frame2

        return sample


def opticalFlowDense(frame1, frame2):
    """
    input: frame1, frame2 (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of frame2
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, pyr_scale = 0.5, levels = 1, winsize = 21,
                                        iterations = 2, poly_n = 7, poly_sigma = 1.5, flags = 10)
    flow = flow.astype(np.float64)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    return rgb

def opticalFlowSparse(frame1, frame2):
    """
    TODO

    input: frame1, frame2 (RGB images)
    calculates sparse optical flow
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255


    curr_pts, _st, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame, self.prev_pts, None, **self.lk_params)
    # Store flow (x, y, dx, dy)
    flow = np.hstack((self.prev_pts.reshape(-1, 2), (curr_pts - self.prev_pts).reshape(-1, 2)))


    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    return rgb


class RICAP(object):
    """
    Inspired by RICAP:
    http://proceedings.mlr.press/v95/takahashi18a/takahashi18a.pdf


    Generates a randomly sized mask in a random location for data augmentation.

    Args:
        beta: (float) alpha and beta term used in beta distribution to generate location of mask
        mask_percentage: (tuple of float) (x, y) where x is lowest percentage and y is highest percentage possible

    """
    def __init__(self, beta = None, mask_percentage = None):
        """

        """
        if beta is None:
            beta = .4
        if mask_percentage is None:
            mask_percentage = (.1, .2)

        self.beta = beta
        self.mask_percentage = mask_percentage

    def __call__(self, image):
        # size of image
        I_y, I_x, _ = image.shape

        # generate boundary position (w, h)
        x1 = int(np.round(I_x * np.random.beta(self.beta, self.beta)))
        y1 = int(np.round(I_y * np.random.beta(self.beta, self.beta)))

        #Get proportional mask rectangle min and max vals
        r = I_y / I_x
        w_min = np.sqrt(self.mask_percentage[0] * I_y * I_x / r)
        l_min = r * w_min
        w_max = np.sqrt(self.mask_percentage[1] * I_y * I_x / r)
        l_max = r * w_max

        w1 = int(np.random.uniform(low=w_min, high=w_max, size=None))
        h1 = int(np.random.uniform(low=l_min, high=l_max, size=None))

        cropper = np.zeros_like(image[y1:min(I_y, y1 + h1), x1:min(I_x, x1 + w1), :])

        image[y1:min(I_y, y1 + h1), x1:min(I_x, x1 + w1), :] = cropper
        image[y1:min(I_y, y1 + h1), x1:min(I_x, x1 + w1), :] = cropper

        return image


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image):
        """
        Convert ndarrays in sample to Tensors.
        """
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).type(self.dtype)


class ToNumpy(object):
    """
    Convert image Tensor arrays to np arrays
    """
    def __call__(self, tensor):
        """
        Convert ndarrays in sample to Tensors.
        """
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return tensor.cpu().permute(1, 2, 0).numpy()
