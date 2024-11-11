import os
import torch
import torchvision
import random
import numpy as np
from sklearn.decomposition import PCA

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.npy']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# Can extract even numpy files now
def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)

# fixed hflip vflip and rot90 
def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[:, :, ::-1]
        if rot90:
            img = img.transpose(0, 2, 1)
        return img

    return [_augment(img) for img in img_list]



def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    # img = torch.from_numpy(np.ascontiguousarray(
    #     np.transpose(img, (2, 0, 1)))).float()

    k = np.clip(img, 0, 1)
    
    img = torch.tensor(k, dtype=torch.float32)
    
    # to range min_max
    min_val = img.min()
    max_val = img.max()
    
    if min_val == max_val:
        return torch.ones_like(img)

    eps = 1e-8
    img = (img-min_val)*(min_max[1] - min_max[0]) / (max_val - min_val) + min_max[0] + eps

    return torch.clip(img,0,1)

totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [transform2tensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0) # stacking to perform the same op on all images
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0) # unstacking to get back the images
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    print("transform augment shape: ", ret_img[0].shape)
    return ret_img


def perform_pca(image_tensor, n_components=3):
    # print("PERFORM PCA SHAPE: ", image_tensor.shape)
    
    # Check if the input is a torch.Tensor, if not, convert it
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.tensor(image_tensor)
    
    if image_tensor.dim() == 3:
        bands, height, width = image_tensor.shape
        batch_size = 1
    elif image_tensor.dim() == 4:
        batch_size, bands, height, width = image_tensor.shape
    else:
        raise ValueError("Input tensor must be 3D (single image) or 4D (batch of images)")
    
    reshaped_image = image_tensor.view(batch_size, bands, -1).permute(0, 2, 1)
    
    reshaped_image_np = reshaped_image.cpu().numpy()
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(reshaped_image_np.reshape(-1, bands))
    
    principal_component_bands = principal_components.reshape(batch_size, height, width, n_components)
    
    principal_component_bands = torch.tensor(principal_component_bands).permute(0, 3, 1, 2)
    
    if batch_size == 1:
        principal_component_bands = principal_component_bands.squeeze(0)
    
    return principal_component_bands