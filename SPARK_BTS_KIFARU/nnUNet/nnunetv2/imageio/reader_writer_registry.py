import traceback
from typing import Type

from batchgenerators.utilities.file_and_folder_operations import join

import nnunetv2
from nnunetv2.imageio.natural_image_reager_writer import NaturalImage2DIO
from nnunetv2.imageio.nibabel_reader_writer import NibabelIO, NibabelIOWithReorient
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.imageio.tif_reader_writer import Tiff3DIO
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import json

LIST_OF_IO_CLASSES = [
    NaturalImage2DIO,
    SimpleITKIO,
    Tiff3DIO,
    NibabelIO,
    NibabelIOWithReorient
]


def determine_reader_writer_from_dataset_json(dataset_json_content: dict, example_file: str = None,
                                              allow_nonmatching_filename: bool = False, verbose: bool = True
                                              ) -> Type[BaseReaderWriter]:
    if 'overwrite_image_reader_writer' in dataset_json_content.keys() and \
            dataset_json_content['overwrite_image_reader_writer'] != 'None':
        ioclass_name = dataset_json_content['overwrite_image_reader_writer']
        # trying to find that class in the nnunetv2.imageio module
        try:
            ret = recursive_find_reader_writer_by_name(ioclass_name)
            if verbose: print('Using %s reader/writer' % ret)
            return ret
        except RuntimeError:
            if verbose: print('Warning: Unable to find ioclass specified in dataset.json: %s' % ioclass_name)
            if verbose: print('Trying to automatically determine desired class')
    return determine_reader_writer_from_file_ending(dataset_json_content['file_ending'], example_file,
                                                    allow_nonmatching_filename, verbose)

def determine_reader_writer_from_file_ending(file_ending: str, example_file: str = None, allow_nonmatching_filename: bool = False,
                                             verbose: bool = True):
    for rw in LIST_OF_IO_CLASSES:
        if file_ending.lower() in rw.supported_file_endings:
            if example_file is not None:
                # if an example file is provided, try if we can actually read it. If not move on to the next reader
                try:
                    tmp = rw()
                    _ = tmp.read_images((example_file,))
                    if verbose: print('Using %s as reader/writer' % rw)
                    return rw
                except:
                    if verbose: print(f'Failed to open file {example_file} with reader {rw}:')
                    traceback.print_exc()
                    pass
            else:
                if verbose: print('Using %s as reader/writer' % rw)
                return rw
        else:
            if allow_nonmatching_filename and example_file is not None:
                try:
                    tmp = rw()
                    _ = tmp.read_images((example_file,))
                    if verbose: print('Using %s as reader/writer' % rw)
                    return rw
                except:
                    if verbose: print(f'Failed to open file {example_file} with reader {rw}:')
                    if verbose: traceback.print_exc()
                    pass
    raise RuntimeError("Unable to determine a reader for file ending %s and file %s (file None means no file provided)." % (file_ending, example_file))


def recursive_find_reader_writer_by_name(rw_class_name: str) -> Type[BaseReaderWriter]:
    ret = recursive_find_python_class(join(nnunetv2.__path__[0], "imageio"), rw_class_name, 'nnunetv2.imageio')
    if ret is None:
        raise RuntimeError("Unable to find reader writer class '%s'. Please make sure this class is located in the "
                           "nnunetv2.imageio module." % rw_class_name)
    else:
        return ret

# import os

# with open('/scratch/guest189/BraTS2023_data/BraTS_Africa_data/nnUNet_raw_data_base/nnUNet_raw/BraTS2023_africa_train_update/Dataset500_BraTS2023/dataset.json', 'r') as json_file:
#     dataset_json = json.load(json_file)
# print(determine_reader_writer_from_dataset_json(dataset_json))

# # Print the dataset JSON content
# print("Dataset JSON Content:")
# print(json.dumps(dataset_json, indent=2))

# # Check for the 'images' key and its values
# if 'images' in dataset_json:
#     images = dataset_json['images']
#     print("List of Images:")
#     print(images)
# else:
#     print("Images key not found in the dataset JSON!")

# # Check if the file paths in 'images' exist
# for image_path in images:
#     if not os.path.exists(image_path):
#         print(f"File not found: {image_path}")

# Determine the reader and writer class for the dataset
# try:
#     reader_writer_class = determine_reader_writer_from_dataset_json(dataset_json, images[0])
#     print("Using reader/writer class:", reader_writer_class)
# except IndexError:
#     print("Error: IndexError occurred. List index out of range.")