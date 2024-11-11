from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d
import numpy as np
import numpy as np
import math

def PSNR(pred, gt):
    valid = gt - pred
    rmse = math.sqrt(np.mean(valid ** 2))   
    
    if rmse == 0:
        return 100
    psnr = 20 * math.log10(1.0 / rmse)
    return psnr   


# def SSIM(img1, img2):
#     batch_size, channels, height, width = img1.shape
#     total_ssim = 0
    
#     for b in range(batch_size):
#         for c in range(channels):
#             channel_ssim = ssim(img1[b, c, :, :], img2[b, c, :, :], data_range=img2[b, c, :, :].max() - img2[b, c, :, :].min())
#             total_ssim += channel_ssim
    
#     average_ssim = total_ssim / (batch_size * channels)
#     return average_ssim 

def SSIM(pred, gt):
    # Initialize SSIM
    ssim = 0
    
    # Iterate over the batch size
    for i in range(gt.shape[0]):  # gt.shape[0] is batch size
        for c in range(gt.shape[1]):  # gt.shape[1] is number of channels
            ssim += compute_ssim(pred[i, c, :, :], gt[i, c, :, :])
    
    # Average over the batch size and number of channels
    return ssim / (gt.shape[0] * gt.shape[1])   	


def normalize(numpy_image, min_max = (-1,1)):
    eps = 2.2204e-16
    min_val = np.min(numpy_image)
    max_val = np.max(numpy_image)
    if min_val == max_val:
        return np.ones_like(numpy_image)
    
    img = numpy_image
    img = (img-min_val)*(min_max[1] - min_max[0]) / (max_val - min_val) + min_max[0] + eps
    return img

def SAM(pred, gt):

    eps = 2.2204e-16
    pred[np.where(pred==0)] = eps
    gt[np.where(gt==0)] = eps 

    # pred = normalize(pred)
    # gt = normalize(gt)
      
    nom = sum(pred*gt)
    denom1 = sum(pred*pred)**0.5
    denom2 = sum(gt*gt)**0.5
    cos_theta = (nom).astype(np.float32)/(denom1*denom2+eps)
    cos_theta = normalize(cos_theta)

    sam = np.real(np.arccos(np.clip(cos_theta, -1, 1)))

    sam[np.isnan(sam)]=0     
    sam_sum = np.mean(sam)*180/np.pi 
      	       
    return  sam_sum



def matlab_style_gauss2D(shape=np.array([11,11]),sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    siz = (shape-np.array([1,1]))/2
    std = sigma
    eps = 2.2204e-16
    x = np.arange(-siz[1], siz[1]+1, 1)
    y = np.arange(-siz[0], siz[1]+1, 1)
    m,n = np.meshgrid(x, y)
    
    h = np.exp(-(m*m + n*n).astype(np.float32) / (2.*sigma*sigma))    
    h[ h < eps*h.max() ] = 0    	
    sumh = h.sum()   	

    if sumh != 0:
        h = h.astype(np.float32) / sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
 
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):
 
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
 
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=np.array([win_size,win_size]), sigma=1.5)
    window = window.astype(np.float32)/np.sum(np.sum(window))
 
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
 
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)).astype(np.float32) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
 
    return np.mean(np.mean(ssim_map))
 
        
