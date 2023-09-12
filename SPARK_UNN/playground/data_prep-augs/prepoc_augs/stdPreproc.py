import numpy as np
from medpy.filter.binary import largest_connected_component
from skimage.exposure import rescale_intensity
from skimage.transform import resize

def dsc(y_pred, y_true, lcc=True):
    if lcc and np.any(y_pred):
        y_pred = np.round(y_pred).astype(int)
        y_true = np.round(y_true).astype(int)
        y_pred = largest_connected_component(y_pred)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

# 1. cropping
def crop_sample(x):
  volume,mask = x
  volume[volume< np.max(volume)*0.1] =0
  #calculating z_proj
  z_proj = np.max(np.max(np.max(volume, axis=-1),axis=-1),axis=-1)
  z_nonzero = np.nonzero(z_proj)
  z_min = np.min(z_nonzero)
  z_max = np.max(z_nonzero)+1
  #calculating y_proj
  y_proj = np.max(np.max(np.max(volume, axis=0),axis=-1),axis=-1)
  y_nonzero = np.nonzero(y_proj)
  y_min = np.min(y_nonzero)
  y_max = np.max(y_nonzero)+1
  #calculating x_proj
  x_proj = np.max(np.max(np.max(volume, axis=0),axis=0),axis=-1)
  x_nonzero = np.nonzero(x_proj)
  x_min = np.min(x_nonzero)
  x_max = np.max(x_nonzero)+1
  volume_cropped = volume[z_min:z_max,y_min:y_max,x_min:x_max]
  mask_cropped = mask[z_min:z_max,y_min:y_max,x_min:x_max]

  return volume_cropped, mask_cropped

#2. padding
def pad_sample(x):
  volume, mask=x
  a = volume.shape[1]
  b= volume.shape[2]
  if a == b:
    return volume,mask
  diff = (max(a,b) - min(a,b))/2.0
  if a>b:
    padding =((0,0),(0,0),(int(np.floor(diff)),int(np.ceil(diff))))
  else:
    padding=((0,0),(int(np.floor(diff)),int(np.ceil(diff))),(0,0))
  mask = np.pad(mask,padding, mode='constant', constant_values=0)
  padding = padding+((0,0),)
  volume = np.pad(volume, padding,mode='constant', constant_values=0)

  return volume, mask

@staticmethod
def pad(image, padding):
    pad_d, pad_w, pad_h = padding
    return np.pad(
        image,
        (
            (0, 0),
            (math.floor(pad_d), math.ceil(pad_d)),
            (math.floor(pad_w), math.ceil(pad_w)),
            (math.floor(pad_h), math.ceil(pad_h)),
        ),
    )

#3. Resizing

def resize_sample(x,size=256):
  volume,mask=x
  v_shape = volume.shape
  out_shape = (v_shape[0],size,size)
  mask = resize(
      mask, output_shape=out_shape,
      order=0,
      mode='constant',
      cval=0,
      anti_aliasing= False,
  )
  out_shape = out_shape + (v_shape[3],)
  volume = resize(
      volume,
      output_shape = out_shape,
      order=2,
      mode='constant',
      cval=0,
      anti_aliasing = False,
  )
  return volume, mask

# 4. Normalizing
def normalize_volume(volume):
  p10 = np.percentile(volume,10)
  p99 = np.percentile(volume,99)
  volume = rescale_intensity(volume, in_range=(p10,p99))
  mean = np.mean(volume, axis=(0,1,2))
  standard_dev = np.std(volume, axis=(0,1,2))
  volume = (volume - mean)/standard_dev
  return volume

class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (max(std, 1e-8))
        return image

