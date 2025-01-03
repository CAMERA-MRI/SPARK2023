# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils.utils import get_task_code
# from preprocessor import Preprocessor
import os
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# data_preprocessing_path = "/home/guest189/Optimized U-Net/Optimized U-Net/data_preprocessing"
# sys.path.append(data_preprocessing_path)
# from data_preprocessing.preprocessor import Preprocessor

# while in the path: /home/guest189/Optimized U-Net/Optimized U-Net/data_preprocessing

data_preprocessing_path = "/home/guest189/SPARK_Stater/Optimized U-Net_BTS/data_preprocessing"
sys.path.append(data_preprocessing_path)
# # # import preprocessor
# import sys
# correct_preprocessor_path = "/home/guest189/Optimized U-Net/Optimized U-Net"
# sys.path.append(correct_preprocessor_path)


# from data_preprocessing.preprocessor import Preprocessor

# # while in path '/home/guest189/Optimized U-Net/Optimized U-Net'
# sys.path.append('data_preprocessing')
# # from data_preprocessing.preprocessor import Preprocessor
from preprocessor import Preprocessor


# from data_preprocessing.preprocessor import Preprocessor
untils_path = "/home/guest189/SPARK_Stater/Optimized U-Net_BTS/utils/utils"
sys.path.append(untils_path)
# from utils import get_task_code

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", type=str,
                    default="/scratch/guest189/BraTS2023_data/BraTS_Africa_data", help="Path to data directory")
parser.add_argument("--results", type=str, default="/scratch/guest189/BraTS2023_data/BraTS_Africa_data",
                    help="Path for saving results directory")
parser.add_argument(
    "--exec_mode",
    type=str,
    default="training",
    choices=["training", "val", "test"],
    help="Mode for data preprocessing",
)
parser.add_argument("--ohe", action="store_true",
                    help="Add one-hot-encoding for foreground voxels (voxels > 0)")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--task", type=str,
                    help="Number of task to be run. MSD uses numbers 01-10")
parser.add_argument("--dim", type=int, default=3,
                    choices=[2, 3], help="Data dimension to prepare")
parser.add_argument("--n_jobs", type=int, default=-1,
                    help="Number of parallel jobs for data preprocessing")


if __name__ == "__main__":
    args = parser.parse_args()
    start = time.time()
    Preprocessor(args).run()
    task_code = get_task_code(args)
    path = os.path.join(args.data, task_code)
    if args.exec_mode == "test":
        path = os.path.join(path, "test")
    end = time.time()
    print(f"Pre-processing time: {(end - start):.2f}")


# python3 '/home/guest189/SPARK_Stater/Optimized U-Net_BTS/preprocess.py' --task 11 --ohe --exec_mode training

# python3 '/home/guest189/SPARK_Stater/Optimized U-Net_BTS/preprocess.py' --task 12 --ohe --exec_mode test
