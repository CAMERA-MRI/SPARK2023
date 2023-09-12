# Mixed Structure Regularisation: corrupt the input data by randomly adding different brain images from BraTS training set, as described by \cite{atya_non_2021}. 

# We can represent this data augmentation with the function: \begin{equation*}MSR(x) = (1 - \alpha ) * x + \alpha * {x_r}\tag{5}\end{equation*}. 
# where x is the original image, and xr is a randomly selected image. 
# This is applied with probability and magnitude of P=0.5, $\alpha$=1∗10−4.


# Shuffle Pixel Noise: A random permutation of the pixels is chosen and then the same permutation is applied to all the images in the training set \cite{atya_non_2021}: 

# We can represent this as \begin{equation*}SPN(x) = (1 - \alpha ) * x + \alpha * {x_r}\tag{6}\end{equation*}, 
# where x is the original image, and xr is the image after shuffling pixels on the x, y axis. 
# This is applied with probability and magnitude: P=1, $\alpha$=1∗10−7.




#1. Rotation class
class Rotate(object):
  def __init__(self,angle):
    self.angle = angle

  def __call__(self,sample):
    image,mask = sample
    angle = np.random.uniform(low=-self.angle, high= self.angle)
    image = rotate(image, angle,resize=False, preserve_range=True, mode="constant")
    mask= rotate(mask,angle,resize=False,order=0, preserve_range=True, mode="constant")
    return image, mask

#2. Horizontal flip class
class HorizontalFlip(object):
  def __init__(self,flip_prob):
    self.flip_prob = flip_prob
  
  def __call__(self,sample):
    image,mask = sample

    if np.random.rand()>self.flip_prob:
      return image,mask
    
    image = np.fliplr(image).copy()
    mask = np.fliplr(mask).copy()

    return image,mask

#applying transform using all the helper function
def transforms(angle=None, flip_prob=None):
  transforms_list = []
  if angle is not None:
    transforms_list.append(Rotate(angle))
  if flip_prob is not None:
    transforms_list.append(HorizontalFlip(flip_prob))
  
  return Compose(transforms_list)

#######################################################################
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
aug_gen_args = dict(shear_range = 0.2,
                    zoom_range = 0.2,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='reflect'
                   )

X_train_gen = ImageDataGenerator(**aug_gen_args)
y_train_gen = ImageDataGenerator(**aug_gen_args)
X_val_gen = ImageDataGenerator()
y_val_gen = ImageDataGenerator()

aug_image_real = X_train[5].reshape((1,)+X_train[1].shape)
aug_image_seg = Y_train[5].reshape((1,)+Y_train[1].shape)
aug_image_real_check = X_train_gen.flow(aug_image_real, batch_size=1, seed=17, shuffle=False)
aug_image_seg_check = y_train_gen.flow(aug_image_seg, batch_size=1, seed=17, shuffle=False)
########################################################

class TrainPipeline(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, **kwargs):
        super().__init__(batch_size, num_threads, device_id, **kwargs)
        self.oversampling = kwargs["oversampling"]
        self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
        self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)

    @staticmethod
    def slice_fn(img):
        return fn.slice(img, 1, 3, axes=[0])

    def resize(self, data, interp_type):
        return fn.resize(data, interp_type=interp_type, size=self.crop_shape_float)

    def biased_crop_fn(self, img, label):
        roi_start, roi_end = fn.segmentation.random_object_bbox(
            label,
            device="cpu",
            background=0,
            format="start_end",
            cache_objects=True,
            foreground_prob=self.oversampling,
        )
        anchor = fn.roi_random_crop(label, roi_start=roi_start, roi_end=roi_end, crop_shape=[1, *self.patch_size])
        anchor = fn.slice(anchor, 1, 3, axes=[0])  # drop channels from anchor
        img, label = fn.slice(
            [img, label], anchor, self.crop_shape, axis_names="DHW", out_of_bounds_policy="pad", device="cpu"
        )
        return img.gpu(), label.gpu()

    def zoom_fn(self, img, lbl):
        scale = random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.0)), 1.0)
        d, h, w = [scale * x for x in self.patch_size]
        if self.dim == 2:
            d = self.patch_size[0]
        img, lbl = fn.crop(img, crop_h=h, crop_w=w, crop_d=d), fn.crop(lbl, crop_h=h, crop_w=w, crop_d=d)
        img, lbl = self.resize(img, types.DALIInterpType.INTERP_CUBIC), self.resize(lbl, types.DALIInterpType.INTERP_NN)
        return img, lbl

    def noise_fn(self, img):
        img_noised = img + fn.random.normal(img, stddev=fn.random.uniform(range=(0.0, 0.33)))
        return random_augmentation(0.15, img_noised, img)

    def blur_fn(self, img):
        img_blurred = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.5, 1.5)))
        return random_augmentation(0.15, img_blurred, img)

    def brightness_fn(self, img):
        brightness_scale = random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    def contrast_fn(self, img):
        scale = random_augmentation(0.15, fn.random.uniform(range=(0.65, 1.5)), 1.0)
        return math.clamp(img * scale, fn.reductions.min(img), fn.reductions.max(img))

    def flips_fn(self, img, lbl):
        kwargs = {
            "horizontal": fn.random.coin_flip(probability=0.5),
            "vertical": fn.random.coin_flip(probability=0.5),
        }
        if self.dim == 3:
            kwargs.update({"depthwise": fn.random.coin_flip(probability=0.5)})
        return fn.flip(img, **kwargs), fn.flip(lbl, **kwargs)

    def define_graph(self):
        img, lbl = self.load_data()
        img, lbl = self.biased_crop_fn(img, lbl)
        img, lbl = self.zoom_fn(img, lbl)
        img, lbl = self.flips_fn(img, lbl)
        img = self.noise_fn(img)
        img = self.blur_fn(img)
        img = self.brightness_fn(img)
        img = self.contrast_fn(img)
        if self.dim == 2:
            img, lbl = self.transpose_fn(img, lbl)
        if self.layout == "NDHWC" and self.dim == 3:
            img, lbl = self.make_dhwc_layout(img, lbl)
        return img, lbl