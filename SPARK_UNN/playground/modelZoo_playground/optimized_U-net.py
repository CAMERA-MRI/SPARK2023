"""
    Optimized U-Net for Brain Tumor Segmentation Model Implementation
    Paper url: https://arxiv.org/pdf/2110.03352.pdf
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import (Conv3D, Conv3DTranspose, Concatenate, LeakyReLU, MaxPooling3D, Input)


def _encoder(pl, num_fil, is_first_conv=False):
    """
    Encoder private function that only applies Conv3D, Instance Normalization and LeakyReLU
    :param pl: previous layer
    :param num_fil: number of filters in Conv3D layer
    :param is_first_conv: whether this conv block is the first one or not
    :return: tensor
    """
    fcls = (2, 2, 2)    # first convolutional layer strides
    scls = (1, 1, 1)    # second convolutional layer strides
    # first conv block condition
    if is_first_conv:
        fcls = (1, 1, 1)

    c_1 = Conv3D(filters=num_fil, kernel_size=(3, 3, 3), strides=fcls, padding="same")(pl)
    in_1 = InstanceNormalization()(c_1)
    lr_1 = LeakyReLU(alpha=0.01)(in_1)  # Activation layer
    c_2 = Conv3D(filters=num_fil, kernel_size=(3, 3, 3), strides=scls, padding="same")(lr_1)
    in_2 = InstanceNormalization()(c_2)
    lr_2 = LeakyReLU(alpha=0.01)(in_2)  # Activation layer
    return lr_2


def encoder(pl, num_fil, is_first_conv=False, is_base=False):
    """
    Encoder function that applies downsampling over two previous convolutional layers
    :param pl: previous layer
    :param num_fil: number of filters in conv3d layer
    :param is_first_conv: whether this conv block is the first one or not
    :param is_base: default is False
    :return: tensor, tensor
    """
    # Conv block return value
    cbrv = _encoder(pl, num_fil, is_first_conv)
    if is_base is True:
        return cbrv
    # Adding downsampling layer
    mp = MaxPooling3D(strides=(2, 2, 2), padding="same")(cbrv)
    return cbrv, mp


def _decoder(pl, num_fil):
    """
    Private decoder function that only applies Conv3D, Instance Normalization and LeakyReLU
    :param pl: previous layer
    :param num_fil: number of filters in conv3d layer
    :return: tensor
    """
    c_1 = Conv3D(filters=num_fil, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(pl)
    in_1 = InstanceNormalization()(c_1)
    lr_1 = LeakyReLU(alpha=0.01)(in_1)
    c_2 = Conv3D(filters=num_fil, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(lr_1)
    in_2 = InstanceNormalization()(c_2)
    lr_2 = LeakyReLU(alpha=0.01)(in_2)
    return lr_2


def decoder(pl, num_fil, has_skip_connection=False, connection=None, has_output=False):
    """
    Decoder function that applies Conv3DTranspose and Concatenation with encoder block convolutional layers
    :param pl: previous layer
    :param num_fil: number of filters in conv3d and conv3dTranspose layer
    :param has_skip_connection: whether has concatenation layer or not
    :param connection: encoder convolutional block return value to concatenate
    :param has_output: whether has output channels with sigmoid activation or not
    :return: tensor
    """
    ct = Conv3DTranspose(filters=num_fil, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same")(pl)
    if has_skip_connection is True and connection is not None:
        con = Concatenate(axis=3)([ct, connection])
        ct = con
    c = _decoder(ct, num_fil)
    if has_output:
        out = Conv3D(filters=num_fil, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding="same", activation=sigmoid)(c)
        c = out
    return c


# Defining input layer
in_layer = Input(shape=(5, 128, 128, 128))

# Defining encoding path blocks
e_1, m_1 = encoder(pl=in_layer, num_fil=64, is_first_conv=True)
e_2, m_2 = encoder(pl=m_1, num_fil=96)
e_3, m_3 = encoder(pl=m_2, num_fil=128)
e_4, m_4 = encoder(pl=m_3, num_fil=192)
e_5, m_5 = encoder(pl=m_4, num_fil=256)
e_6, m_6 = encoder(pl=m_5, num_fil=384)

# Defining the base
base = encoder(pl=m_6, num_fil=512, is_base=True)

# Defining decoding path blocks
d_1 = decoder(pl=base, num_fil=384)
d_2 = decoder(pl=d_1, num_fil=265, has_skip_connection=True, connection=e_3)
d_3 = decoder(pl=d_2, num_fil=192, has_skip_connection=True, connection=e_4)
d_4 = decoder(pl=d_3, num_fil=128, has_skip_connection=True, connection=e_3, has_output=True)
d_5 = decoder(pl=d_4, num_fil=96, has_skip_connection=True, connection=e_2, has_output=True)
d_6 = decoder(pl=d_5, num_fil=64, has_skip_connection=True, connection=e_1, has_output=True)


# Creating the model
model = Model(inputs=[in_layer], outputs=[d_4, d_5, d_6], name="Optimized_U_Net")


if __name__ == '__main__':
    # Displaying model architecture details
    model.summary()
