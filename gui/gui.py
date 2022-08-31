import sys
import time
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QComboBox, QApplication, QHBoxLayout, QMainWindow, QPushButton, QSpinBox, QVBoxLayout, QWidget, QFileDialog, QLabel, QErrorMessage
from PyQt5.QtGui import QFont, QImage, QPixmap, QIcon
import cv2
import numpy as np
from collections import Counter

import argparse
import os
import math
from functools import partial

import codecs
from typing import Any, Dict, Generic

import glob
import platform
import subprocess

import inspect
from collections.abc import Sequence
import warnings

import paddle
import yaml

import contextlib
import filelock
import tempfile
import random
from urllib.parse import urlparse, unquote

import functools
import shutil
import tarfile
import zipfile
import requests


from PIL import Image as PILImage
from PIL import ImageEnhance
from scipy.ndimage.morphology import distance_transform_edt


import collections.abc
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as paddle_init


class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)

    def hasHeightForWidth(self):
        return self.pixmap() is not None

    def heightForWidth(self, w):
        if self.pixmap():
            try:
                return int(w * (self.pixmap().height() / self.pixmap().width()))
            except ZeroDivisionError:
                return 0


def resize_image(image_data, max_img_width, max_img_height):
    scale_percent = min(max_img_width / image_data.shape[1], max_img_height / image_data.shape[0])
    width = int(image_data.shape[1] * scale_percent)
    height = int(image_data.shape[0] * scale_percent)
    newSize = (width, height)
    image_resized = cv2.resize(image_data, newSize, None, None, None, cv2.INTER_AREA)
    return image_resized


def pixmap_from_cv_image(cv_image):
    height, width, _ = cv_image.shape
    bytesPerLine = 3 * width
    qImg = QImage(cv_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()
    return QPixmap(qImg)


def get_image_mask(img, blue_shade, diff):
    lower = np.array([max(0, val) for val in [blue_shade[2]-diff, blue_shade[1]-diff, blue_shade[0]-diff]], dtype=np.uint8)
    upper = np.array([min(255, val) for val in [blue_shade[2]+diff, blue_shade[1]+diff, blue_shade[0]+diff]], dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)
    percent_traced = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return cv2.cvtColor(mask, cv2.COLOR_BGR2RGB), percent_traced




def parse_args():
    
    parser = argparse.ArgumentParser(description='Model prediction')

    # params of prediction
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help=
        'The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./output/result')

    # augment for prediction
    parser.add_argument(
        '--aug_pred',
        dest='aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        dest='scales',
        nargs='+',
        help='Scales for augment',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        dest='flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        dest='flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # sliding window prediction
    parser.add_argument(
        '--is_slide',
        dest='is_slide',
        help='Whether to prediction by sliding window',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        dest='crop_size',
        nargs=2,
        help=
        'The crop size of sliding window, the first is width and the second is height.',
        type=int,
        default=None)
    parser.add_argument(
        '--stride',
        dest='stride',
        nargs=2,
        help=
        'The stride of sliding window, the first is width and the second is height.',
        type=int,
        default=None)

    # custom color map
    parser.add_argument(
        '--custom_color',
        dest='custom_color',
        nargs='+',
        help=
        'Save images with a custom color map. Default: None, use paddleseg\'s default color map.',
        type=int,
        default=None)
    return parser


"""$utils/env/sys_env"""

IS_WINDOWS = sys.platform == 'win32'

def _find_cuda_home():
    '''Finds the CUDA install path. It refers to the implementation of
    pytorch <https://github.com/pytorch/pytorch/blob/master/torch/utils/cpp_extension.py>.
    '''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
            nvcc = subprocess.check_output([which,
                                            'nvcc']).decode().rstrip('\r\n')
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    return cuda_home

def _get_nvcc_info(cuda_home):
    if cuda_home is not None and os.path.isdir(cuda_home):
        try:
            nvcc = os.path.join(cuda_home, 'bin/nvcc')
            nvcc = subprocess.check_output(
                "{} -V".format(nvcc), shell=True).decode()
            nvcc = nvcc.strip().split('\n')[-1]
        except subprocess.SubprocessError:
            nvcc = "Not Available"
    else:
        nvcc = "Not Available"
    return nvcc

def _get_gpu_info():
    try:
        gpu_info = subprocess.check_output(['nvidia-smi',
                                            '-L']).decode().strip()
        gpu_info = gpu_info.split('\n')
        for i in range(len(gpu_info)):
            gpu_info[i] = ' '.join(gpu_info[i].split(' ')[:4])
    except:
        gpu_info = ' Can not get GPU information. Please make sure CUDA have been installed successfully.'
    return gpu_info

def get_sys_env():
    """collect environment information"""
    env_info = {}
    env_info['platform'] = platform.platform()

    env_info['Python'] = sys.version.replace('\n', '')

    # TODO is_compiled_with_cuda() has not been moved
    compiled_with_cuda = paddle.is_compiled_with_cuda()
    env_info['Paddle compiled with cuda'] = compiled_with_cuda

    if compiled_with_cuda:
        cuda_home = _find_cuda_home()
        env_info['NVCC'] = _get_nvcc_info(cuda_home)
        # refer to https://github.com/PaddlePaddle/Paddle/blob/release/2.0-rc/paddle/fluid/platform/device_context.cc#L327
        v = paddle.get_cudnn_version()
        v = str(v // 1000) + '.' + str(v % 1000 // 100)
        env_info['cudnn'] = v
        if 'gpu' in paddle.get_device():
            gpu_nums = paddle.distributed.ParallelEnv().nranks
        else:
            gpu_nums = 0
        env_info['GPUs used'] = gpu_nums

        env_info['CUDA_VISIBLE_DEVICES'] = os.environ.get(
            'CUDA_VISIBLE_DEVICES')
        if gpu_nums == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        env_info['GPU'] = _get_gpu_info()

    try:
        gcc = subprocess.check_output(['gcc', '--version']).decode()
        gcc = gcc.strip().split('\n')[0]
        env_info['GCC'] = gcc
    except:
        pass

    '''
    env_info['PaddleSeg'] = paddleseg.__version__
    env_info['PaddlePaddle'] = paddle.__version__
    '''
    env_info['OpenCV'] = cv2.__version__
    
    return env_info




'''$cvlibs/config'''

class Config(object):


    def __init__(self,
                 path: str,
                 learning_rate: float = None,
                 batch_size: int = None,
                 iters: int = None):
        if not path:
            raise ValueError('Please specify the configuration file path.')

        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))

        self._model = None
        self._losses = None
        if path.endswith('yml') or path.endswith('yaml'):
            self.dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError('Config file should in yaml format!')

        self.update(
            learning_rate=learning_rate, batch_size=batch_size, iters=iters)

    def _update_dic(self, dic, base_dic):
        """
        Update config from dic based base_dic
        """
        base_dic = base_dic.copy()
        dic = dic.copy()

        if dic.get('_inherited_', True) == False:
            dic.pop('_inherited_')
            return dic

        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = self._update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def _parse_from_yaml(self, path: str):
        '''Parse a yaml file and build config'''
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        if '_base_' in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop('_base_')
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = self._parse_from_yaml(base_path)
            dic = self._update_dic(dic, base_dic)
        return dic

    def update(self,
               learning_rate: float = None,
               batch_size: int = None,
               iters: int = None):
        '''Update config'''
        if learning_rate:
            if 'lr_scheduler' in self.dic:
                self.dic['lr_scheduler']['learning_rate'] = learning_rate
            else:
                self.dic['learning_rate']['value'] = learning_rate

        if batch_size:
            self.dic['batch_size'] = batch_size

        if iters:
            self.dic['iters'] = iters

    @property
    def batch_size(self) -> int:
        return self.dic.get('batch_size', 1)

    @property
    def iters(self) -> int:
        iters = self.dic.get('iters')
        if not iters:
            raise RuntimeError('No iters specified in the configuration file.')
        return iters

    @property
    def lr_scheduler(self) -> paddle.optimizer.lr.LRScheduler:
        if 'lr_scheduler' not in self.dic:
            raise RuntimeError(
                'No `lr_scheduler` specified in the configuration file.')
        params = self.dic.get('lr_scheduler')

        lr_type = params.pop('type')
        if lr_type == 'PolynomialDecay':
            params.setdefault('decay_steps', self.iters)
            params.setdefault('end_lr', 0)
            params.setdefault('power', 0.9)

        return getattr(paddle.optimizer.lr, lr_type)(**params)

    @property
    def learning_rate(self) -> paddle.optimizer.lr.LRScheduler:
        warning(
            '''`learning_rate` in configuration file will be deprecated, please use `lr_scheduler` instead. E.g
            lr_scheduler:
                type: PolynomialDecay
                learning_rate: 0.01''')

        _learning_rate = self.dic.get('learning_rate', {})
        if isinstance(_learning_rate, float):
            return _learning_rate

        _learning_rate = self.dic.get('learning_rate', {}).get('value')
        if not _learning_rate:
            raise RuntimeError(
                'No learning rate specified in the configuration file.')

        args = self.decay_args
        decay_type = args.pop('type')

        if decay_type == 'poly':
            lr = _learning_rate
            return paddle.optimizer.lr.PolynomialDecay(lr, **args)
        elif decay_type == 'piecewise':
            values = _learning_rate
            return paddle.optimizer.lr.PiecewiseDecay(values=values, **args)
        elif decay_type == 'stepdecay':
            lr = _learning_rate
            return paddle.optimizer.lr.StepDecay(lr, **args)
        else:
            raise RuntimeError('Only poly and piecewise decay support.')

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        if 'lr_scheduler' in self.dic:
            lr = self.lr_scheduler
        else:
            lr = self.learning_rate
        args = self.optimizer_args
        optimizer_type = args.pop('type')

        if optimizer_type == 'sgd':
            return paddle.optimizer.Momentum(
                lr, parameters=self.model.parameters(), **args)
        elif optimizer_type == 'adam':
            return paddle.optimizer.Adam(
                lr, parameters=self.model.parameters(), **args)
        elif optimizer_type in paddle.optimizer.__all__:
            return getattr(paddle.optimizer, optimizer_type)(
                lr, parameters=self.model.parameters(), **args)

        raise RuntimeError('Unknown optimizer type {}.'.format(optimizer_type))

    @property
    def optimizer_args(self) -> dict:
        args = self.dic.get('optimizer', {}).copy()
        if args['type'] == 'sgd':
            args.setdefault('momentum', 0.9)

        return args

    @property
    def decay_args(self) -> dict:
        args = self.dic.get('learning_rate', {}).get('decay', {
            'type': 'poly',
            'power': 0.9
        }).copy()

        if args['type'] == 'poly':
            args.setdefault('decay_steps', self.iters)
            args.setdefault('end_lr', 0)

        return args

    @property
    def loss(self) -> dict:
        if self._losses is None:
            self._losses = self._prepare_loss('loss')
        return self._losses

    @property
    def distill_loss(self) -> dict:
        if not hasattr(self, '_distill_losses'):
            self._distill_losses = self._prepare_loss('distill_loss')
        return self._distill_losses

    def _prepare_loss(self, loss_name):
        """
        Parse the loss parameters and load the loss layers.

        Args:
            loss_name (str): The root name of loss in the yaml file.
        Returns:
            dict: A dict including the loss parameters and layers.
        """
        args = self.dic.get(loss_name, {}).copy()
        if 'types' in args and 'coef' in args:
            len_types = len(args['types'])
            len_coef = len(args['coef'])
            if len_types != len_coef:
                if len_types == 1:
                    args['types'] = args['types'] * len_coef
                else:
                    raise ValueError(
                        'The length of types should equal to coef or equal to 1 in loss config, but they are {} and {}.'
                        .format(len_types, len_coef))
        else:
            raise ValueError(
                'Loss config should contain keys of "types" and "coef"')

        losses = dict()
        for key, val in args.items():
            if key == 'types':
                losses['types'] = []
                for item in args['types']:
                    if item['type'] != 'MixedLoss':
                        if 'ignore_index' in item:
                            assert item['ignore_index'] == self.train_dataset.ignore_index, 'If ignore_index of loss is set, '\
                            'the ignore_index of loss and train_dataset must be the same. \nCurrently, loss ignore_index = {}, '\
                            'train_dataset ignore_index = {}. \nIt is recommended not to set loss ignore_index, so it is consistent with '\
                            'train_dataset by default.'.format(item['ignore_index'], self.train_dataset.ignore_index)
                        item['ignore_index'] = \
                            self.train_dataset.ignore_index
                    losses['types'].append(self._load_object(item))
            else:
                losses[key] = val
        if len(losses['coef']) != len(losses['types']):
            raise RuntimeError(
                'The length of coef should equal to types in loss config: {} != {}.'
                .format(len(losses['coef']), len(losses['types'])))
        return losses

    @property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.dic.get('model').copy()
        if not model_cfg:
            raise RuntimeError('No model specified in the configuration file.')
        if not 'num_classes' in model_cfg:
            num_classes = None
            if self.train_dataset_config:
                if hasattr(self.train_dataset_class, 'NUM_CLASSES'):
                    num_classes = self.train_dataset_class.NUM_CLASSES
                elif hasattr(self.train_dataset, 'num_classes'):
                    num_classes = self.train_dataset.num_classes
            elif self.val_dataset_config:
                if hasattr(self.val_dataset_class, 'NUM_CLASSES'):
                    num_classes = self.val_dataset_class.NUM_CLASSES
                elif hasattr(self.val_dataset, 'num_classes'):
                    num_classes = self.val_dataset.num_classes

            if num_classes is not None:
                model_cfg['num_classes'] = num_classes

        if not self._model:
            self._model = self._load_object(model_cfg)
        return self._model

    @property
    def train_dataset_config(self) -> Dict:
        return self.dic.get('train_dataset', {}).copy()

    @property
    def val_dataset_config(self) -> Dict:
        return self.dic.get('val_dataset', {}).copy()

    @property
    def train_dataset_class(self) -> Generic:
        dataset_type = self.train_dataset_config['type']
        return self._load_component(dataset_type)

    @property
    def val_dataset_class(self) -> Generic:
        dataset_type = self.val_dataset_config['type']
        return self._load_component(dataset_type)

    @property
    def train_dataset(self) -> paddle.io.Dataset:
        _train_dataset = self.train_dataset_config
        if not _train_dataset:
            return None
        return self._load_object(_train_dataset)

    @property
    def val_dataset(self) -> paddle.io.Dataset:
        _val_dataset = self.val_dataset_config
        if not _val_dataset:
            return None
        return self._load_object(_val_dataset)

    def _load_component(self, com_name: str) -> Any:
        com_list = [
            MODELS, BACKBONES, DATASETS,
            TRANSFORMS, LOSSES
        ]

        for com in com_list:
            if com_name in com.components_dict:
                return com[com_name]
        else:
            raise RuntimeError(
                'The specified component was not found {}.'.format(com_name))

    def _load_object(self, cfg: dict) -> Any:
        cfg = cfg.copy()
        if 'type' not in cfg:
            raise RuntimeError('No object information in {}.'.format(cfg))

        component = self._load_component(cfg.pop('type'))

        params = {}
        for key, val in cfg.items():
            if self._is_meta_type(val):
                params[key] = self._load_object(val)
            elif isinstance(val, list):
                params[key] = [
                    self._load_object(item)
                    if self._is_meta_type(item) else item for item in val
                ]
            else:
                params[key] = val

        return component(**params)

    @property
    def test_config(self) -> Dict:
        return self.dic.get('test_config', {})

    @property
    def export_config(self) -> Dict:
        return self.dic.get('export', {})

    @property
    def to_static_training(self) -> bool:
        '''Whether to use @to_static for training'''
        return self.dic.get('to_static_training', False)

    def _is_meta_type(self, item: Any) -> bool:
        return isinstance(item, dict) and 'type' in item

    def __str__(self) -> str:
        return yaml.dump(self.dic)





'''$utils/logger'''

levels = {0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}
log_level = 2

def log(level=2, message=""):
    if paddle.distributed.ParallelEnv().local_rank == 0:
        current_time = time.time()
        time_array = time.localtime(current_time)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
        if log_level >= level:
            print(
                "{} [{}]\t{}".format(current_time, levels[level],
                                     message).encode("utf-8").decode("latin1"))
            sys.stdout.flush()

def debug(message=""):
    log(level=3, message=message)

def info(message=""):
    log(level=2, message=message)

def warning(message=""):
    log(level=1, message=message)

def error(message=""):
    log(level=0, message=message)





'''$cvlibs/manager'''


class ComponentManager:

    def __init__(self, name=None):
        self._components_dict = dict()
        self._name = name

    def __len__(self):
        return len(self._components_dict)

    def __repr__(self):
        name_str = self._name if self._name else self.__class__.__name__
        return "{}:{}".format(name_str, list(self._components_dict.keys()))

    def __getitem__(self, item):
        if item not in self._components_dict.keys():
            raise KeyError("{} does not exist in availabel {}".format(
                item, self))
        return self._components_dict[item]

    @property
    def components_dict(self):
        return self._components_dict

    @property
    def name(self):
        return self._name

    def _add_single_component(self, component):

        # Currently only support class or function type
        if not (inspect.isclass(component) or inspect.isfunction(component)):
            raise TypeError(
                "Expect class/function type, but received {}".format(
                    type(component)))

        # Obtain the internal name of the component
        component_name = component.__name__

        # Check whether the component was added already
        if component_name in self._components_dict.keys():
            warnings.warn(
                "{} exists already! It is now updated to {} !!!".format(
                    component_name, component))
            self._components_dict[component_name] = component

        else:
            # Take the internal name of the component as its key
            self._components_dict[component_name] = component

    def add_component(self, components):

        # Check whether the type is a sequence
        if isinstance(components, Sequence):
            for component in components:
                self._add_single_component(component)
        else:
            component = components
            self._add_single_component(component)

        return components

MODELS = ComponentManager("models")
BACKBONES = ComponentManager("backbones")
DATASETS = ComponentManager("datasets")
TRANSFORMS = ComponentManager("transforms")
LOSSES = ComponentManager("losses")





'''$datasets/dataset'''

@DATASETS.add_component
class Dataset(paddle.io.Dataset):

    def __init__(self,
                 transforms,
                 dataset_root,
                 num_classes,
                 mode='train',
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.edge = edge

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    self.mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))

        if self.mode == 'train':
            if train_path is None:
                raise ValueError(
                    'When `mode` is "train", `train_path` is necessary, but it is None.'
                )
            elif not os.path.exists(train_path):
                raise FileNotFoundError(
                    '`train_path` is not found: {}'.format(train_path))
            else:
                file_path = train_path
        elif self.mode == 'val':
            if val_path is None:
                raise ValueError(
                    'When `mode` is "val", `val_path` is necessary, but it is None.'
                )
            elif not os.path.exists(val_path):
                raise FileNotFoundError(
                    '`val_path` is not found: {}'.format(val_path))
            else:
                file_path = val_path
        else:
            if test_path is None:
                raise ValueError(
                    'When `mode` is "test", `test_path` is necessary, but it is None.'
                )
            elif not os.path.exists(test_path):
                raise FileNotFoundError(
                    '`test_path` is not found: {}'.format(test_path))
            else:
                file_path = test_path

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) != 2:
                    if self.mode == 'train' or self.mode == 'val':
                        raise ValueError(
                            "File list format incorrect! In training or evaluation task it should be"
                            " image_name{}label_name\\n".format(separator))
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = os.path.join(self.dataset_root, items[1])
                self.file_list.append([image_path, label_path])

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'val':
            im, _ = self.transforms(im=image_path)
            label = np.asarray(PILImage.open(label_path))
            label = label[np.newaxis, :, :]
            return im, label
        else:
            im, label = self.transforms(im=image_path, label=label_path)
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                return im, label, edge_mask
            else:
                return im, label

    def __len__(self):
        return len(self.file_list)




'''$utils/env/seg_env'''
def _get_user_home():
    return os.path.expanduser('~')

def _get_seg_home():
    if 'SEG_HOME' in os.environ:
        home_path = os.environ['SEG_HOME']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                warning('SEG_HOME {} is a file!'.format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.paddleseg')

def _get_sub_home(directory):
    home = os.path.join(_get_seg_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home

USER_HOME = _get_user_home()
SEG_HOME = _get_seg_home()
DATA_HOME = _get_sub_home('dataset')
TMP_HOME = _get_sub_home('tmp')
PRETRAINED_MODEL_HOME = _get_sub_home('pretrained_model')



'''$models/layers/layer_libs'''
def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    elif paddle.distributed.ParallelEnv().nranks == 1:
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._relu = Activation("relu")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class ConvBNAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 act_type=None,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)

        self._act_type = act_type
        if act_type is not None:
            self._act = Activation(act_type)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        if self._act_type is not None:
            x = self._act(x)
        return x


class ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvReLUPool(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)
        self._relu = Activation("relu")
        self._max_pool = nn.MaxPool2D(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self._relu(x)
        x = self._max_pool(x)
        return x


class SeparableConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 pointwise_bias=None,
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self.piontwise_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            data_format=data_format,
            bias_attr=pointwise_bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


class AuxLayer(nn.Layer):
    """
    The auxiliary layer implementation for auxiliary loss.
    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1,
                 **kwargs):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            **kwargs)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2D(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class JPU(nn.Layer):
    """
    Joint Pyramid Upsampling of FCN.
    The original paper refers to
        Wu, Huikai, et al. "Fastfcn: Rethinking dilated convolution in the backbone for semantic segmentation." arXiv preprint arXiv:1903.11816 (2019).
    """

    def __init__(self, in_channels, width=512):
        super().__init__()

        self.conv5 = ConvBNReLU(
            in_channels[-1], width, 3, padding=1, bias_attr=False)
        self.conv4 = ConvBNReLU(
            in_channels[-2], width, 3, padding=1, bias_attr=False)
        self.conv3 = ConvBNReLU(
            in_channels[-3], width, 3, padding=1, bias_attr=False)

        self.dilation1 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=1,
            pointwise_bias=False,
            dilation=1,
            bias_attr=False,
            stride=1, )
        self.dilation2 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=2,
            pointwise_bias=False,
            dilation=2,
            bias_attr=False,
            stride=1)
        self.dilation3 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=4,
            pointwise_bias=False,
            dilation=4,
            bias_attr=False,
            stride=1)
        self.dilation4 = SeparableConvBNReLU(
            3 * width,
            width,
            3,
            padding=8,
            pointwise_bias=False,
            dilation=8,
            bias_attr=False,
            stride=1)

    def forward(self, *inputs):
        feats = [
            self.conv5(inputs[-1]), self.conv4(inputs[-2]),
            self.conv3(inputs[-3])
        ]
        size = paddle.shape(feats[-1])[2:]
        feats[-2] = F.interpolate(
            feats[-2], size, mode='bilinear', align_corners=True)
        feats[-3] = F.interpolate(
            feats[-3], size, mode='bilinear', align_corners=True)

        feat = paddle.concat(feats, axis=1)
        feat = paddle.concat(
            [
                self.dilation1(feat), self.dilation2(feat),
                self.dilation3(feat), self.dilation4(feat)
            ],
            axis=1)

        return inputs[0], inputs[1], inputs[2], feat


class ConvBNPReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._prelu = Activation("prelu")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._prelu(x)
        return x


class ConvBNLeakyReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        self._relu = Activation("leakyrelu")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return 


'''$models/layers/pyramid_pool'''

class ASPPModule(nn.Layer):

    def __init__(self,
                 aspp_ratios,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_sep_conv=False,
                 image_pooling=False,
                 data_format='NCHW'):
        super().__init__()

        self.align_corners = align_corners
        self.data_format = data_format
        self.aspp_blocks = nn.LayerList()

        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_func = SeparableConvBNReLU
            else:
                conv_func = ConvBNReLU

            block = conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio,
                data_format=data_format)
            self.aspp_blocks.append(block)

        out_size = len(self.aspp_blocks)

        if image_pooling:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2D(
                    output_size=(1, 1), data_format=data_format),
                ConvBNReLU(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias_attr=False,
                    data_format=data_format))
            out_size += 1
        self.image_pooling = image_pooling

        self.conv_bn_relu = ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1,
            data_format=data_format)

        self.dropout = nn.Dropout(p=0.1)  # drop rate

    def forward(self, x):
        outputs = []
        if self.data_format == 'NCHW':
            interpolate_shape = paddle.shape(x)[2:]
            axis = 1
        else:
            interpolate_shape = paddle.shape(x)[1:3]
            axis = -1
        for block in self.aspp_blocks:
            y = block(x)
            outputs.append(y)

        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            img_avg = F.interpolate(
                img_avg,
                interpolate_shape,
                mode='bilinear',
                align_corners=self.align_corners,
                data_format=self.data_format)
            outputs.append(img_avg)

        x = paddle.concat(outputs, axis=axis)
        x = self.conv_bn_relu(x)
        x = self.dropout(x)

        return x


class PPModule(nn.Layer):
    """
    Pyramid pooling module originally in PSPNet.

    Args:
        in_channels (int): The number of intput channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 2, 3, 6).
        dim_reduction (bool, optional): A bool value represents if reducing dimension after pooling. Default: True.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self, in_channels, out_channels, bin_sizes, dim_reduction,
                 align_corners):
        super().__init__()

        self.bin_sizes = bin_sizes

        inter_channels = in_channels
        if dim_reduction:
            inter_channels = in_channels // len(bin_sizes)

        # we use dimension reduction after pooling mentioned in original implementation.
        self.stages = nn.LayerList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_bn_relu2 = ConvBNReLU(
            in_channels=in_channels + inter_channels * len(bin_sizes),
            out_channels=out_channels,
            kernel_size=3,
            padding=1)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        """
        Create one pooling layer.

        In our implementation, we adopt the same dimension reduction as the original paper that might be
        slightly different with other implementations.

        After pooling, the channels are reduced to 1/len(bin_sizes) immediately, while some other implementations
        keep the channels to be same.

        Args:
            in_channels (int): The number of intput channels to pyramid pooling module.
            size (int): The out size of the pooled layer.

        Returns:
            conv (Tensor): A tensor after Pyramid Pooling Module.
        """

        prior = nn.AdaptiveAvgPool2D(output_size=(size, size))
        conv = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        return nn.Sequential(prior, conv)

    def forward(self, input):
        cat_layers = []
        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                paddle.shape(input)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            cat_layers.append(x)
        cat_layers = [input] + cat_layers[::-1]
        cat = paddle.concat(cat_layers, axis=1)
        out = self.conv_bn_relu2(cat)

        return out





'''$models/layers/activation'''

class Activation(nn.Layer):

    def __init__(self, act=None):
        super(Activation, self).__init__()

        self._act = act
        upper_act_names = nn.layer.activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))

        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                self.act_func = eval(
                    "nn.layer.activation.{}()".format(act_name))
            else:
                raise KeyError("{} does not exist in the current {}".format(
                    act, act_dict.keys()))

    def forward(self, x):
        if self._act is not None:
            return self.act_func(x)
        else:
            return x




'''$cvlibs/param_init'''


def constant_init(param, **kwargs):

    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)


def normal_init(param, **kwargs):

    initializer = nn.initializer.Normal(**kwargs)
    initializer(param, param.block)


def kaiming_normal_init(param, **kwargs):

    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)


def kaiming_uniform(param, **kwargs):


    initializer = nn.initializer.KaimingUniform(**kwargs)
    initializer(param, param.block)



'''$transforms/functional'''


def normalize(im, mean, std):
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def resize(im, target_size=608, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


def resize_long(im, long_size=224, interpolation=cv2.INTER_LINEAR):
    value = max(im.shape[0], im.shape[1])
    scale = float(long_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(
        im, (resized_width, resized_height), interpolation=interpolation)
    return im


def resize_short(im, short_size=224, interpolation=cv2.INTER_LINEAR):
    value = min(im.shape[0], im.shape[1])
    scale = float(short_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(
        im, (resized_width, resized_height), interpolation=interpolation)
    return im


def horizontal_flip(im):
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    return im


def vertical_flip(im):
    if len(im.shape) == 3:
        im = im[::-1, :, :]
    elif len(im.shape) == 2:
        im = im[::-1, :]
    return im


def brightness(im, brightness_lower, brightness_upper):
    brightness_delta = np.random.uniform(brightness_lower, brightness_upper)
    im = ImageEnhance.Brightness(im).enhance(brightness_delta)
    return im


def contrast(im, contrast_lower, contrast_upper):
    contrast_delta = np.random.uniform(contrast_lower, contrast_upper)
    im = ImageEnhance.Contrast(im).enhance(contrast_delta)
    return im


def saturation(im, saturation_lower, saturation_upper):
    saturation_delta = np.random.uniform(saturation_lower, saturation_upper)
    im = ImageEnhance.Color(im).enhance(saturation_delta)
    return im


def hue(im, hue_lower, hue_upper):
    hue_delta = np.random.uniform(hue_lower, hue_upper)
    im = np.array(im.convert('HSV'))
    im[:, :, 0] = im[:, :, 0] + hue_delta
    im = PILImage.fromarray(im, mode='HSV').convert('RGB')
    return im


def sharpness(im, sharpness_lower, sharpness_upper):
    sharpness_delta = np.random.uniform(sharpness_lower, sharpness_upper)
    im = ImageEnhance.Sharpness(im).enhance(sharpness_delta)
    return im


def rotate(im, rotate_lower, rotate_upper):
    rotate_delta = np.random.uniform(rotate_lower, rotate_upper)
    im = im.rotate(int(rotate_delta))
    return im


def mask_to_onehot(mask, num_classes):

    _mask = [mask == i for i in range(num_classes)]
    _mask = np.array(_mask).astype(np.uint8)
    return _mask


def onehot_to_binary_edge(mask, radius):

    if radius < 1:
        raise ValueError('`radius` should be greater than or equal to 1')
    num_classes = mask.shape[0]

    edge = np.zeros(mask.shape[1:])
    # pad borders
    mask = np.pad(
        mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    for i in range(num_classes):
        dist = distance_transform_edt(
            mask[i, :]) + distance_transform_edt(1.0 - mask[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edge += dist

    edge = np.expand_dims(edge, axis=0)
    edge = (edge > 0).astype(np.uint8)
    return edge


def mask_to_binary_edge(mask, radius, num_classes):

    mask = mask.squeeze()
    onehot = mask_to_onehot(mask, num_classes)
    edge = onehot_to_binary_edge(onehot, radius)
    return edge








'''$models/fcn'''

@MODELS.add_component
class FCN(nn.Layer):

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(-1, ),
                 channels=None,
                 align_corners=False,
                 pretrained=None,
                 bias=True,
                 data_format="NCHW"):
        super(FCN, self).__init__()

        if data_format != 'NCHW':
            raise ('fcn only support NCHW data format')
        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = FCNHead(
            num_classes,
            backbone_indices,
            backbone_channels,
            channels,
            bias=bias)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.data_format = data_format
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)


class FCNHead(nn.Layer):

    def __init__(self,
                 num_classes,
                 backbone_indices=(-1, ),
                 backbone_channels=(270, ),
                 channels=None,
                 bias=True):
        super(FCNHead, self).__init__()

        self.num_classes = num_classes
        self.backbone_indices = backbone_indices
        if channels is None:
            channels = backbone_channels[0]

        self.conv_1 = ConvBNReLU(
            in_channels=backbone_channels[0],
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias_attr=bias)
        self.cls = nn.Conv2D(
            in_channels=channels,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            bias_attr=bias)
        self.init_weight()

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[0]]
        x = self.conv_1(x)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                constant_init(layer.weight, value=1.0)
                constant_init(layer.bias, value=0.0)



'''$models/fastfcn'''

@MODELS.add_component
class FastFCN(nn.Layer):


    def __init__(self,
                 num_classes,
                 backbone,
                 num_codes=32,
                 mid_channels=512,
                 use_jpu=True,
                 aux_loss=True,
                 use_se_loss=True,
                 add_lateral=False,
                 pretrained=None):
        super().__init__()
        self.add_lateral = add_lateral
        self.num_codes = num_codes
        self.backbone = backbone
        self.use_jpu = use_jpu
        in_channels = self.backbone.feat_channels

        if use_jpu:
            self.jpu_layer = JPU(in_channels, mid_channels)
            in_channels[-1] = mid_channels * 4
            self.bottleneck = ConvBNReLU(
                in_channels[-1],
                mid_channels,
                1,
                padding=0,
                bias_attr=False, )
        else:
            self.bottleneck = ConvBNReLU(
                in_channels[-1],
                mid_channels,
                3,
                padding=1,
                bias_attr=False, )
        if self.add_lateral:
            self.lateral_convs = nn.LayerList([
                ConvBNReLU(
                    in_channels[0], mid_channels, 1, bias_attr=False),
                ConvBNReLU(
                    in_channels[1], mid_channels, 1, bias_attr=False),
            ])

            self.fusion = ConvBNReLU(
                3 * mid_channels,
                mid_channels,
                3,
                padding=1,
                bias_attr=False, )

        self.enc_module = EncModule(mid_channels, num_codes)
        self.cls_seg = nn.Conv2D(mid_channels, num_classes, 1)

        self.aux_loss = aux_loss
        if self.aux_loss:
            self.fcn_head = AuxLayer(in_channels[-2], mid_channels,
                                            num_classes)

        self.use_se_loss = use_se_loss
        if use_se_loss:
            self.se_layer = nn.Linear(mid_channels, num_classes)

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)

    def forward(self, inputs):
        imsize = paddle.shape(inputs)[2:]
        feats = self.backbone(inputs)
        if self.use_jpu:
            feats = self.jpu_layer(*feats)

        fcn_feat = feats[2]

        feat = self.bottleneck(feats[-1])
        if self.add_lateral:
            laterals = []
            for i, lateral_conv in enumerate(self.lateral_convs):
                laterals.append(
                    F.interpolate(
                        lateral_conv(feats[i]),
                        size=paddle.shape(feat)[2:],
                        mode='bilinear',
                        align_corners=False))
            feat = self.fusion(paddle.concat([feat, *laterals], 1))
        encode_feat, feat = self.enc_module(feat)
        out = self.cls_seg(feat)
        out = F.interpolate(
            out, size=imsize, mode='bilinear', align_corners=False)
        output = [out]

        if self.training:
            fcn_out = self.fcn_head(fcn_feat)
            fcn_out = F.interpolate(
                fcn_out, size=imsize, mode='bilinear', align_corners=False)
            output.append(fcn_out)
            if self.use_se_loss:
                se_out = self.se_layer(encode_feat)
                output.append(se_out)
            return output
        return output


class Encoding(nn.Layer):
    def __init__(self, channels, num_codes):
        super().__init__()
        self.channels, self.num_codes = channels, num_codes

        std = 1 / ((channels * num_codes)**0.5)
        self.codewords = self.create_parameter(
            shape=(num_codes, channels),
            default_initializer=nn.initializer.Uniform(-std, std), )
        self.scale = self.create_parameter(
            shape=(num_codes, ),
            default_initializer=nn.initializer.Uniform(-1, 0), )

    def scaled_l2(self, x, codewords, scale):
        num_codes, channels = paddle.shape(codewords)
        reshaped_scale = scale.reshape([1, 1, num_codes])
        expanded_x = paddle.tile(x.unsqueeze(2), [1, 1, num_codes, 1])
        reshaped_codewords = codewords.reshape([1, 1, num_codes, channels])

        scaled_l2_norm = reshaped_scale * (
            expanded_x - reshaped_codewords).pow(2).sum(axis=3)
        return scaled_l2_norm

    def aggregate(self, assignment_weights, x, codewords):
        num_codes, channels = paddle.shape(codewords)
        reshaped_codewords = codewords.reshape([1, 1, num_codes, channels])
        expanded_x = paddle.tile(
            x.unsqueeze(2),
            [1, 1, num_codes, 1], )
        encoded_feat = (assignment_weights.unsqueeze(3) *
                        (expanded_x - reshaped_codewords)).sum(axis=1)
        return encoded_feat

    def forward(self, x):
        x_dims = x.ndim
        assert x_dims == 4, "The dimension of input tensor must equal 4, but got {}.".format(
            x_dims)
        assert paddle.shape(
            x
        )[1] == self.channels, "Encoding channels error, excepted {} but got {}.".format(
            self.channels, paddle.shape(x)[1])
        batch_size = paddle.shape(x)[0]
        x = x.reshape([batch_size, self.channels, -1]).transpose([0, 2, 1])
        assignment_weights = F.softmax(
            self.scaled_l2(x, self.codewords, self.scale), axis=2)

        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        encoded_feat = encoded_feat.reshape([batch_size, self.num_codes, -1])
        return encoded_feat


class EncModule(nn.Layer):
    def __init__(self, in_channels, num_codes):
        super().__init__()
        self.encoding_project = ConvBNReLU(
            in_channels,
            in_channels,
            1, )
        self.encoding = nn.Sequential(
            Encoding(
                channels=in_channels, num_codes=num_codes),
            nn.BatchNorm1D(num_codes),
            nn.ReLU(), )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid(), )

    def forward(self, x):
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(axis=1)
        batch_size, channels, _, _ = paddle.shape(x)
        gamma = self.fc(encoding_feat)
        y = gamma.reshape([batch_size, channels, 1, 1])
        output = F.relu(x + x * y)
        return encoding_feat, output




'''$models/pp_liteseg'''

@MODELS.add_component
class PPLiteSeg(nn.Layer):


    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[2, 3, 4],
                 arm_type='UAFM_SpAtten',
                 cm_bin_sizes=[1, 2, 4],
                 cm_out_ch=128,
                 arm_out_chs=[64, 96, 128],
                 seg_head_inter_chs=[64, 64, 64],
                 resize_mode='bilinear',
                 pretrained=None):
        super().__init__()

        # backbone
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
            "should be greater than 1"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

        # head
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
            "arm_out_chs and backbone_indices should be equal"

        self.ppseg_head = PPLiteSegHead(backbone_out_chs, arm_out_chs,
                                        cm_bin_sizes, cm_out_ch, arm_type,
                                        resize_mode)

        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        assert len(seg_head_inter_chs) == len(backbone_indices), "The length of " \
            "seg_head_inter_chs and backbone_indices should be equal"
        self.seg_heads = nn.LayerList()  # [..., head_16, head32]
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]

        feats_head = self.ppseg_head(feats_selected)  # [..., x8, x16, x32]

        if self.training:
            logit_list = []

            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)

            logit_list = [
                F.interpolate(
                    x, x_hw, mode='bilinear', align_corners=False)
                for x in logit_list
            ]
        else:
            x = self.seg_heads[0](feats_head[0])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit_list = [x]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)


class PPLiteSegHead(nn.Layer):


    def __init__(self, backbone_out_chs, arm_out_chs, cm_bin_sizes, cm_out_ch,
                 arm_type, resize_mode):
        super().__init__()

        self.cm = PPContextModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch,
                                  cm_bin_sizes)

        #assert hasattr(layers,arm_type), \
        #    "Not support arm_type ({})".format(arm_type)
        arm_class = eval("layers." + arm_type)

        self.arm_list = nn.LayerList()  # [..., arm8, arm16, arm32]
        for i in range(len(backbone_out_chs)):
            low_chs = backbone_out_chs[i]
            high_ch = cm_out_ch if i == len(
                backbone_out_chs) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(
                low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)

    def forward(self, in_feat_list):


        high_feat = self.cm(in_feat_list[-1])
        out_feat_list = []

        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)

        return out_feat_list


class PPContextModule(nn.Layer):


    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super().__init__()

        self.stages = nn.LayerList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = ConvBNReLU(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2D(output_size=size)
        conv = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = paddle.shape(input)[2:]

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(
                x,
                input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x

        out = self.conv_out(out)
        return out


class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(
            in_chan,
            mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.conv_out = nn.Conv2D(
            mid_chan, n_classes, kernel_size=1, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


'''$models/backbones/transformer_utils'''

def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob=0., training=False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):


    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


trunc_normal_ = paddle_init.TruncatedNormal(std=.02)
zeros_ = paddle_init.Constant(value=0.)
ones_ = paddle_init.Constant(value=1.)


def init_weights(layer):

    if isinstance(layer, nn.Linear):
        trunc_normal_(layer.weight)
        if layer.bias is not None:
            zeros_(layer.bias)
    elif isinstance(layer, nn.LayerNorm):
        zeros_(layer.bias)
        ones_(layer.weight)


'''$models/backbones/mix_transformer'''

class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.dim = dim

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x, H, W):
        x_shape = paddle.shape(x)
        B, N = x_shape[0], x_shape[1]
        C = self.dim

        q = self.q(x).reshape([B, N, self.num_heads,
                               C // self.num_heads]).transpose([0, 2, 1, 3])

        if self.sr_ratio > 1:
            x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
            x_ = self.sr(x_).reshape([B, C, -1]).transpose([0, 2, 1])
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(
                [B, -1, 2, self.num_heads,
                 C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        else:
            kv = self.kv(x).reshape(
                [B, -1, 2, self.num_heads,
                 C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]

        attn = (q @k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Layer):


    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x):
        x = self.proj(x)
        x_shape = paddle.shape(x)
        H, W = x_shape[2], x_shape[3]
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Layer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.feat_channels = embed_dims[:]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [
            x.numpy() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.LayerList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.LayerList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.LayerList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.LayerList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            load_pretrained_model(self, self.pretrained)
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            paddle_init.Normal(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item()
            for x in paddle.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim,
                              num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = paddle.shape(x)[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)

        x = self.norm1(x)
        x = x.reshape([B, H, W, self.feat_channels[0]]).transpose([0, 3, 1, 2])
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape([B, H, W, self.feat_channels[1]]).transpose([0, 3, 1, 2])
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape([B, H, W, self.feat_channels[2]]).transpose([0, 3, 1, 2])
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape([B, H, W, self.feat_channels[3]]).transpose([0, 3, 1, 2])
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Layer):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dim = dim
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias_attr=True, groups=dim)

    def forward(self, x, H, W):
        x_shape = paddle.shape(x)
        B, N = x_shape[0], x_shape[1]
        x = x.transpose([0, 2, 1]).reshape([B, self.dim, H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose([0, 2, 1])

        return x


@BACKBONES.add_component
def MixVisionTransformer_B0(**kwargs):
    return MixVisionTransformer(
        patch_size=4,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs)


@BACKBONES.add_component
def MixVisionTransformer_B1(**kwargs):
    return MixVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs)


@BACKBONES.add_component
def MixVisionTransformer_B2(**kwargs):
    return MixVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs)


@BACKBONES.add_component
def MixVisionTransformer_B3(**kwargs):
    return MixVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs)


@BACKBONES.add_component
def MixVisionTransformer_B4(**kwargs):
    return MixVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[3, 8, 27, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs)


@BACKBONES.add_component
def MixVisionTransformer_B5(**kwargs):
    return MixVisionTransformer(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[3, 6, 40, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs)


'''$models/segformer'''


class MLP(nn.Layer):


    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x


@MODELS.add_component
class SegFormer(nn.Layer):


    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim,
                 align_corners=False,
                 pretrained=None):
        super(SegFormer, self).__init__()

        self.pretrained = pretrained
        self.align_corners = align_corners
        self.backbone = backbone
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = ConvBNReLU(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            bias_attr=False)

        self.linear_pred = nn.Conv2D(
            embedding_dim, self.num_classes, kernel_size=1)

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)
        c1, c2, c3, c4 = feats

        ############## MLP decoder on C1-C4 ###########
        c1_shape = paddle.shape(c1)
        c2_shape = paddle.shape(c2)
        c3_shape = paddle.shape(c3)
        c4_shape = paddle.shape(c4)

        _c4 = self.linear_c4(c4).transpose([0, 2, 1]).reshape(
            [0, 0, c4_shape[2], c4_shape[3]])
        _c4 = F.interpolate(
            _c4,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).transpose([0, 2, 1]).reshape(
            [0, 0, c3_shape[2], c3_shape[3]])
        _c3 = F.interpolate(
            _c3,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).transpose([0, 2, 1]).reshape(
            [0, 0, c2_shape[2], c2_shape[3]])
        _c2 = F.interpolate(
            _c2,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).transpose([0, 2, 1]).reshape(
            [0, 0, c1_shape[2], c1_shape[3]])

        _c = self.linear_fuse(paddle.concat([_c4, _c3, _c2, _c1], axis=1))

        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        return [
            F.interpolate(
                logit,
                size=paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]


@MODELS.add_component
def SegFormer_B0(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B0'](),
        embedding_dim=256,
        **kwargs)


@MODELS.add_component
def SegFormer_B1(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B1'](),
        embedding_dim=256,
        **kwargs)


@MODELS.add_component
def SegFormer_B2(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B2'](),
        embedding_dim=768,
        **kwargs)


@MODELS.add_component
def SegFormer_B3(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B3'](),
        embedding_dim=768,
        **kwargs)


@MODELS.add_component
def SegFormer_B4(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B4'](),
        embedding_dim=768,
        **kwargs)


@MODELS.add_component
def SegFormer_B5(**kwargs):
    return SegFormer(
        backbone=BACKBONES['MixVisionTransformer_B5'](),
        embedding_dim=768,
        **kwargs)


'''$models/deeplab'''

@MODELS.add_component
class DeepLabV3P(nn.Layer):


    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(0, 3),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 align_corners=False,
                 pretrained=None,
                 data_format="NCHW"):
        super().__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = DeepLabV3PHead(
            num_classes,
            backbone_indices,
            backbone_channels,
            aspp_ratios,
            aspp_out_channels,
            align_corners,
            data_format=data_format)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.data_format = data_format
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        if self.data_format == 'NCHW':
            ori_shape = paddle.shape(x)[2:]
        else:
            ori_shape = paddle.shape(x)[1:3]
        return [
            F.interpolate(
                logit,
                ori_shape,
                mode='bilinear',
                align_corners=self.align_corners,
                data_format=self.data_format) for logit in logit_list
        ]

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)

class DeepLabV3PHead(nn.Layer):

    def __init__(self,
                 num_classes,
                 backbone_indices,
                 backbone_channels,
                 aspp_ratios,
                 aspp_out_channels,
                 align_corners,
                 data_format='NCHW'):
        super().__init__()

        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels[1],
            aspp_out_channels,
            align_corners,
            use_sep_conv=True,
            image_pooling=True,
            data_format=data_format)
        self.decoder = DecoderDeepLab(
            num_classes,
            backbone_channels[0],
            align_corners,
            data_format=data_format)
        self.backbone_indices = backbone_indices

    def forward(self, feat_list):
        logit_list = []
        low_level_feat = feat_list[self.backbone_indices[0]]
        x = feat_list[self.backbone_indices[1]]
        x = self.aspp(x)
        logit = self.decoder(x, low_level_feat)
        logit_list.append(logit)

        return logit_list


@MODELS.add_component
class DeepLabV3(nn.Layer):

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(3, ),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = DeepLabV3Head(num_classes, backbone_indices,
                                  backbone_channels, aspp_ratios,
                                  aspp_out_channels, align_corners)
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)


class DeepLabV3Head(nn.Layer):


    def __init__(self, num_classes, backbone_indices, backbone_channels,
                 aspp_ratios, aspp_out_channels, align_corners):
        super().__init__()

        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels[0],
            aspp_out_channels,
            align_corners,
            use_sep_conv=False,
            image_pooling=True)

        self.cls = nn.Conv2D(
            in_channels=aspp_out_channels,
            out_channels=num_classes,
            kernel_size=1)

        self.backbone_indices = backbone_indices

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[0]]
        x = self.aspp(x)
        logit = self.cls(x)
        logit_list.append(logit)

        return logit_list


class DecoderDeepLab(nn.Layer):


    def __init__(self,
                 num_classes,
                 in_channels,
                 align_corners,
                 data_format='NCHW'):
        super(DecoderDeepLab, self).__init__()

        self.data_format = data_format
        self.conv_bn_relu1 = ConvBNReLU(
            in_channels=in_channels,
            out_channels=48,
            kernel_size=1,
            data_format=data_format)

        self.conv_bn_relu2 = SeparableConvBNReLU(
            in_channels=304,
            out_channels=256,
            kernel_size=3,
            padding=1,
            data_format=data_format)
        self.conv_bn_relu3 = SeparableConvBNReLU(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1,
            data_format=data_format)
        self.conv = nn.Conv2D(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            data_format=data_format)

        self.align_corners = align_corners

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_bn_relu1(low_level_feat)
        if self.data_format == 'NCHW':
            low_level_shape = paddle.shape(low_level_feat)[-2:]
            axis = 1
        else:
            low_level_shape = paddle.shape(low_level_feat)[1:3]
            axis = -1
        x = F.interpolate(
            x,
            low_level_shape,
            mode='bilinear',
            align_corners=self.align_corners,
            data_format=self.data_format)
        x = paddle.concat([x, low_level_feat], axis=axis)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv(x)
        return x



'''$models/pspnet'''

@MODELS.add_component
class PSPNet(nn.Layer):


    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(2, 3),
                 pp_out_channels=1024,
                 bin_sizes=(1, 2, 3, 6),
                 enable_auxiliary_loss=True,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = PSPNetHead(num_classes, backbone_indices, backbone_channels,
                               pp_out_channels, bin_sizes,
                               enable_auxiliary_loss, align_corners)
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)


class PSPNetHead(nn.Layer):


    def __init__(self, num_classes, backbone_indices, backbone_channels,
                 pp_out_channels, bin_sizes, enable_auxiliary_loss,
                 align_corners):

        super().__init__()

        self.backbone_indices = backbone_indices

        self.psp_module = PPModule(
            in_channels=backbone_channels[1],
            out_channels=pp_out_channels,
            bin_sizes=bin_sizes,
            dim_reduction=True,
            align_corners=align_corners)

        self.dropout = nn.Dropout(p=0.1)  # dropout_prob

        self.conv = nn.Conv2D(
            in_channels=pp_out_channels,
            out_channels=num_classes,
            kernel_size=1)

        if enable_auxiliary_loss:
            self.auxlayer = AuxLayer(
                in_channels=backbone_channels[0],
                inter_channels=backbone_channels[0] // 4,
                out_channels=num_classes)

        self.enable_auxiliary_loss = enable_auxiliary_loss

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[1]]
        x = self.psp_module(x)
        x = self.dropout(x)
        logit = self.conv(x)
        logit_list.append(logit)

        if self.enable_auxiliary_loss:
            auxiliary_feat = feat_list[self.backbone_indices[0]]
            auxiliary_logit = self.auxlayer(auxiliary_feat)
            logit_list.append(auxiliary_logit)

        return logit_list




'''$models/backbones/hrnet'''

class HRNet(nn.Layer):

    def __init__(self,
                 pretrained=None,
                 stage1_num_modules=1,
                 stage1_num_blocks=(4, ),
                 stage1_num_channels=(64, ),
                 stage2_num_modules=1,
                 stage2_num_blocks=(4, 4),
                 stage2_num_channels=(18, 36),
                 stage3_num_modules=4,
                 stage3_num_blocks=(4, 4, 4),
                 stage3_num_channels=(18, 36, 72),
                 stage4_num_modules=3,
                 stage4_num_blocks=(4, 4, 4, 4),
                 stage4_num_channels=(18, 36, 72, 144),
                 has_se=False,
                 align_corners=False,
                 padding_same=True):
        super(HRNet, self).__init__()
        self.pretrained = pretrained
        self.stage1_num_modules = stage1_num_modules
        self.stage1_num_blocks = stage1_num_blocks
        self.stage1_num_channels = stage1_num_channels
        self.stage2_num_modules = stage2_num_modules
        self.stage2_num_blocks = stage2_num_blocks
        self.stage2_num_channels = stage2_num_channels
        self.stage3_num_modules = stage3_num_modules
        self.stage3_num_blocks = stage3_num_blocks
        self.stage3_num_channels = stage3_num_channels
        self.stage4_num_modules = stage4_num_modules
        self.stage4_num_blocks = stage4_num_blocks
        self.stage4_num_channels = stage4_num_channels
        self.has_se = has_se
        self.align_corners = align_corners
        self.feat_channels = [sum(stage4_num_channels)]

        self.conv_layer1_1 = ConvBNReLU(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        self.conv_layer1_2 = ConvBNReLU(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        self.la1 = Layer1(
            num_channels=64,
            num_blocks=self.stage1_num_blocks[0],
            num_filters=self.stage1_num_channels[0],
            has_se=has_se,
            name="layer2",
            padding_same=padding_same)

        self.tr1 = TransitionLayer(
            in_channels=[self.stage1_num_channels[0] * 4],
            out_channels=self.stage2_num_channels,
            name="tr1",
            padding_same=padding_same)

        self.st2 = Stage(
            num_channels=self.stage2_num_channels,
            num_modules=self.stage2_num_modules,
            num_blocks=self.stage2_num_blocks,
            num_filters=self.stage2_num_channels,
            has_se=self.has_se,
            name="st2",
            align_corners=align_corners,
            padding_same=padding_same)

        self.tr2 = TransitionLayer(
            in_channels=self.stage2_num_channels,
            out_channels=self.stage3_num_channels,
            name="tr2",
            padding_same=padding_same)
        self.st3 = Stage(
            num_channels=self.stage3_num_channels,
            num_modules=self.stage3_num_modules,
            num_blocks=self.stage3_num_blocks,
            num_filters=self.stage3_num_channels,
            has_se=self.has_se,
            name="st3",
            align_corners=align_corners,
            padding_same=padding_same)

        self.tr3 = TransitionLayer(
            in_channels=self.stage3_num_channels,
            out_channels=self.stage4_num_channels,
            name="tr3",
            padding_same=padding_same)
        self.st4 = Stage(
            num_channels=self.stage4_num_channels,
            num_modules=self.stage4_num_modules,
            num_blocks=self.stage4_num_blocks,
            num_filters=self.stage4_num_channels,
            has_se=self.has_se,
            name="st4",
            align_corners=align_corners,
            padding_same=padding_same)

        self.init_weight()

    def forward(self, x):
        conv1 = self.conv_layer1_1(x)
        conv2 = self.conv_layer1_2(conv1)

        la1 = self.la1(conv2)

        tr1 = self.tr1([la1])
        st2 = self.st2(tr1)

        tr2 = self.tr2(st2)
        st3 = self.st3(tr2)

        tr3 = self.tr3(st3)
        st4 = self.st4(tr3)

        size = paddle.shape(st4[0])[2:]
        x1 = F.interpolate(
            st4[1], size, mode='bilinear', align_corners=self.align_corners)
        x2 = F.interpolate(
            st4[2], size, mode='bilinear', align_corners=self.align_corners)
        x3 = F.interpolate(
            st4[3], size, mode='bilinear', align_corners=self.align_corners)
        x = paddle.concat([st4[0], x1, x2, x3], axis=1)

        return [x]

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                constant_init(layer.weight, value=1.0)
                constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            load_pretrained_model(self, self.pretrained)



class Layer1(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 num_blocks,
                 has_se=False,
                 name=None,
                 padding_same=True):
        super(Layer1, self).__init__()

        self.bottleneck_block_list = []

        for i in range(num_blocks):
            bottleneck_block = self.add_sublayer(
                "bb_{}_{}".format(name, i + 1),
                BottleneckBlockHRNet(
                    num_channels=num_channels if i == 0 else num_filters * 4,
                    num_filters=num_filters,
                    has_se=has_se,
                    stride=1,
                    downsample=True if i == 0 else False,
                    name=name + '_' + str(i + 1),
                    padding_same=padding_same))
            self.bottleneck_block_list.append(bottleneck_block)

    def forward(self, x):
        conv = x
        for block_func in self.bottleneck_block_list:
            conv = block_func(conv)
        return conv


class TransitionLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, name=None, padding_same=True):
        super(TransitionLayer, self).__init__()

        num_in = len(in_channels)
        num_out = len(out_channels)
        self.conv_bn_func_list = []
        for i in range(num_out):
            residual = None
            if i < num_in:
                if in_channels[i] != out_channels[i]:
                    residual = self.add_sublayer(
                        "transition_{}_layer_{}".format(name, i + 1),
                        ConvBNReLU(
                            in_channels=in_channels[i],
                            out_channels=out_channels[i],
                            kernel_size=3,
                            padding=1 if not padding_same else 'same',
                            bias_attr=False))
            else:
                residual = self.add_sublayer(
                    "transition_{}_layer_{}".format(name, i + 1),
                    ConvBNReLU(
                        in_channels=in_channels[-1],
                        out_channels=out_channels[i],
                        kernel_size=3,
                        stride=2,
                        padding=1 if not padding_same else 'same',
                        bias_attr=False))
            self.conv_bn_func_list.append(residual)

    def forward(self, x):
        outs = []
        for idx, conv_bn_func in enumerate(self.conv_bn_func_list):
            if conv_bn_func is None:
                outs.append(x[idx])
            else:
                if idx < len(x):
                    outs.append(conv_bn_func(x[idx]))
                else:
                    outs.append(conv_bn_func(x[-1]))
        return outs


class Branches(nn.Layer):
    def __init__(self,
                 num_blocks,
                 in_channels,
                 out_channels,
                 has_se=False,
                 name=None,
                 padding_same=True):
        super(Branches, self).__init__()

        self.basic_block_list = []

        for i in range(len(out_channels)):
            self.basic_block_list.append([])
            for j in range(num_blocks[i]):
                in_ch = in_channels[i] if j == 0 else out_channels[i]
                basic_block_func = self.add_sublayer(
                    "bb_{}_branch_layer_{}_{}".format(name, i + 1, j + 1),
                    BasicBlockHRNet(
                        num_channels=in_ch,
                        num_filters=out_channels[i],
                        has_se=has_se,
                        name=name + '_branch_layer_' + str(i + 1) + '_' +
                        str(j + 1),
                        padding_same=padding_same))
                self.basic_block_list[i].append(basic_block_func)

    def forward(self, x):
        outs = []
        for idx, input in enumerate(x):
            conv = input
            for basic_block_func in self.basic_block_list[idx]:
                conv = basic_block_func(conv)
            outs.append(conv)
        return outs


class BottleneckBlockHRNet(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se,
                 stride=1,
                 downsample=False,
                 name=None,
                 padding_same=True):
        super(BottleneckBlockHRNet, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNReLU(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            bias_attr=False)

        self.conv2 = ConvBNReLU(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        self.conv3 = ConvBN(
            in_channels=num_filters,
            out_channels=num_filters * 4,
            kernel_size=1,
            bias_attr=False)

        if self.downsample:
            self.conv_down = ConvBN(
                in_channels=num_channels,
                out_channels=num_filters * 4,
                kernel_size=1,
                bias_attr=False)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16,
                name=name + '_fc')

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv3 = self.se(conv3)

        y = conv3 + residual
        y = F.relu(y)
        return y


class BasicBlockHRNet(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 has_se=False,
                 downsample=False,
                 name=None,
                 padding_same=True):
        super(BasicBlockHRNet, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNReLU(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            padding=1 if not padding_same else 'same',
            bias_attr=False)
        self.conv2 = ConvBN(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1 if not padding_same else 'same',
            bias_attr=False)

        if self.downsample:
            self.conv_down = ConvBNReLU(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=1,
                bias_attr=False)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16,
                name=name + '_fc')

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv2 = self.se(conv2)

        y = conv2 + residual
        y = F.relu(y)
        return y


class SELayer(nn.Layer):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = nn.AdaptiveAvgPool2D(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = nn.Linear(
            num_channels,
            med_ch,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = nn.Linear(
            med_ch,
            num_filters,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, x):
        pool = self.pool2d_gap(x)
        pool = paddle.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = F.sigmoid(excitation)
        excitation = paddle.reshape(
            excitation, shape=[-1, self._num_channels, 1, 1])
        out = x * excitation
        return out


class Stage(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_modules,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False,
                 padding_same=True):
        super(Stage, self).__init__()

        self._num_modules = num_modules

        self.stage_func_list = []
        for i in range(num_modules):
            if i == num_modules - 1 and not multi_scale_output:
                stage_func = self.add_sublayer(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        multi_scale_output=False,
                        name=name + '_' + str(i + 1),
                        align_corners=align_corners,
                        padding_same=padding_same))
            else:
                stage_func = self.add_sublayer(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        name=name + '_' + str(i + 1),
                        align_corners=align_corners,
                        padding_same=padding_same))

            self.stage_func_list.append(stage_func)

    def forward(self, x):
        out = x
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out)
        return out


class HighResolutionModule(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False,
                 padding_same=True):
        super(HighResolutionModule, self).__init__()

        self.branches_func = Branches(
            num_blocks=num_blocks,
            in_channels=num_channels,
            out_channels=num_filters,
            has_se=has_se,
            name=name,
            padding_same=padding_same)

        self.fuse_func = FuseLayers(
            in_channels=num_filters,
            out_channels=num_filters,
            multi_scale_output=multi_scale_output,
            name=name,
            align_corners=align_corners,
            padding_same=padding_same)

    def forward(self, x):
        out = self.branches_func(x)
        out = self.fuse_func(out)
        return out


class FuseLayers(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False,
                 padding_same=True):
        super(FuseLayers, self).__init__()

        self._actual_ch = len(in_channels) if multi_scale_output else 1
        self._in_channels = in_channels
        self.align_corners = align_corners

        self.residual_func_list = []
        for i in range(self._actual_ch):
            for j in range(len(in_channels)):
                if j > i:
                    residual_func = self.add_sublayer(
                        "residual_{}_layer_{}_{}".format(name, i + 1, j + 1),
                        ConvBN(
                            in_channels=in_channels[j],
                            out_channels=out_channels[i],
                            kernel_size=1,
                            bias_attr=False))
                    self.residual_func_list.append(residual_func)
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBN(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[i],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1 if not padding_same else 'same',
                                    bias_attr=False))
                            pre_num_filters = out_channels[i]
                        else:
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBNReLU(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[j],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1 if not padding_same else 'same',
                                    bias_attr=False))
                            pre_num_filters = out_channels[j]
                        self.residual_func_list.append(residual_func)

    def forward(self, x):
        outs = []
        residual_func_idx = 0
        for i in range(self._actual_ch):
            residual = x[i]
            residual_shape = paddle.shape(residual)[-2:]
            for j in range(len(self._in_channels)):
                if j > i:
                    y = self.residual_func_list[residual_func_idx](x[j])
                    residual_func_idx += 1

                    y = F.interpolate(
                        y,
                        residual_shape,
                        mode='bilinear',
                        align_corners=self.align_corners)
                    residual = residual + y
                elif j < i:
                    y = x[j]
                    for k in range(i - j):
                        y = self.residual_func_list[residual_func_idx](y)
                        residual_func_idx += 1

                    residual = residual + y

            residual = F.relu(residual)
            outs.append(residual)

        return outs


@BACKBONES.add_component
def HRNet_W18(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[18, 36],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[18, 36, 72],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[18, 36, 72, 144],
        **kwargs)
    return model

@BACKBONES.add_component
def HRNet_W48(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[48, 96],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[48, 96, 192],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[48, 96, 192, 384],
        **kwargs)
    return model




'''$models/backbones/resnet_vd'''

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 is_vd_mode=False,
                 act=None,
                 data_format='NCHW'):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2,
            stride=2,
            padding=0,
            ceil_mode=True,
            data_format=data_format)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 if dilation == 1 else 0,
            dilation=dilation,
            groups=groups,
            bias_attr=False,
            data_format=data_format)

        self._batch_norm = SyncBatchNorm(
            out_channels, data_format=data_format)
        self._act_op = Activation(act=act)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self._act_op(y)

        return y


class BottleneckBlockResNet(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 dilation=1,
                 data_format='NCHW'):
        super(BottleneckBlockResNet, self).__init__()

        self.data_format = data_format
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            data_format=data_format)

        self.dilation = dilation

        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            dilation=dilation,
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            data_format=data_format)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True,
                data_format=data_format)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)

        ####################################################################
        # If given dilation rate > 1, using corresponding padding.
        # The performance drops down without the follow padding.
        if self.dilation > 1:
            padding = self.dilation
            y = F.pad(
                y, [padding, padding, padding, padding],
                data_format=self.data_format)
        #####################################################################

        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class BasicBlockResNet(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 data_format='NCHW'):
        super(BasicBlockResNet, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            data_format=data_format)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                data_format=data_format)

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv1)
        y = F.relu(y)

        return y


class ResNet_vd(nn.Layer):


    def __init__(self,
                 layers=50,
                 output_stride=8,
                 multi_grid=(1, 1, 1),
                 pretrained=None,
                 data_format='NCHW'):
        super(ResNet_vd, self).__init__()

        self.data_format = data_format
        self.conv1_logit = None  # for gscnn shape stream
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024
                        ] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        # for channels of four returned stages
        self.feat_channels = [c * 4 for c in num_filters
                              ] if layers >= 50 else num_filters

        dilation_dict = None
        if output_stride == 8:
            dilation_dict = {2: 2, 3: 4}
        elif output_stride == 16:
            dilation_dict = {3: 2}

        self.conv1_1 = ConvBNLayer(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act='relu',
            data_format=data_format)
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            data_format=data_format)
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            data_format=data_format)
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, data_format=data_format)

        # self.block_list = []
        self.stage_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)

                    ###############################################################################
                    # Add dilation rate for some segmentation tasks, if dilation_dict is not None.
                    dilation_rate = dilation_dict[
                        block] if dilation_dict and block in dilation_dict else 1

                    # Actually block here is 'stage', and i is 'block' in 'stage'
                    # At the stage 4, expand the the dilation_rate if given multi_grid
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]
                    ###############################################################################

                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlockResNet(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0
                            and dilation_rate == 1 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            dilation=dilation_rate,
                            data_format=data_format))

                    block_list.append(bottleneck_block)
                    shortcut = True
                self.stage_list.append(block_list)
        else:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlockResNet(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            data_format=data_format))
                    block_list.append(basic_block)
                    shortcut = True
                self.stage_list.append(block_list)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        self.conv1_logit = y.clone()
        y = self.pool2d_max(y)

        # A feature list saves the output feature map of each stage.
        feat_list = []
        for stage in self.stage_list:
            for block in stage:
                y = block(y)
            feat_list.append(y)

        return feat_list

    def init_weight(self):
        load_pretrained_model(self, self.pretrained)

@BACKBONES.add_component
def ResNet50_vd(**args):
    model = ResNet_vd(layers=50, **args)
    return model


@BACKBONES.add_component
def ResNet101_vd(**args):
    model = ResNet_vd(layers=101, **args)
    return model






'''$models/segnet'''

@MODELS.add_component
class SegNet(nn.Layer):


    def __init__(self, num_classes, pretrained=None):
        super().__init__()

        # Encoder Module

        self.enco1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, padding=1),
            ConvBNReLU(64, 64, 3, padding=1))

        self.enco2 = nn.Sequential(
            ConvBNReLU(64, 128, 3, padding=1),
            ConvBNReLU(128, 128, 3, padding=1))

        self.enco3 = nn.Sequential(
            ConvBNReLU(128, 256, 3, padding=1),
            ConvBNReLU(256, 256, 3, padding=1),
            ConvBNReLU(256, 256, 3, padding=1))

        self.enco4 = nn.Sequential(
            ConvBNReLU(256, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1))

        self.enco5 = nn.Sequential(
            ConvBNReLU(512, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1))

        # Decoder Module

        self.deco1 = nn.Sequential(
            ConvBNReLU(512, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1))

        self.deco2 = nn.Sequential(
            ConvBNReLU(512, 512, 3, padding=1),
            ConvBNReLU(512, 512, 3, padding=1),
            ConvBNReLU(512, 256, 3, padding=1))

        self.deco3 = nn.Sequential(
            ConvBNReLU(256, 256, 3, padding=1),
            ConvBNReLU(256, 256, 3, padding=1),
            ConvBNReLU(256, 128, 3, padding=1))

        self.deco4 = nn.Sequential(
            ConvBNReLU(128, 128, 3, padding=1),
            ConvBNReLU(128, 128, 3, padding=1),
            ConvBNReLU(128, 64, 3, padding=1))

        self.deco5 = nn.Sequential(
            ConvBNReLU(64, 64, 3, padding=1),
            nn.Conv2D(64, num_classes, kernel_size=3, padding=1),
        )

        self.pretrained = pretrained

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)

    def forward(self, x):
        logit_list = []

        x = self.enco1(x)
        x, ind1 = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        size1 = x.shape

        x = self.enco2(x)
        x, ind2 = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        size2 = x.shape

        x = self.enco3(x)
        x, ind3 = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        size3 = x.shape

        x = self.enco4(x)
        x, ind4 = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        size4 = x.shape

        x = self.enco5(x)
        x, ind5 = F.max_pool2d(x, kernel_size=2, stride=2, return_mask=True)
        size5 = x.shape

        x = F.max_unpool2d(
            x, indices=ind5, kernel_size=2, stride=2, output_size=size4)
        x = self.deco1(x)

        x = F.max_unpool2d(
            x, indices=ind4, kernel_size=2, stride=2, output_size=size3)
        x = self.deco2(x)

        x = F.max_unpool2d(
            x, indices=ind3, kernel_size=2, stride=2, output_size=size2)
        x = self.deco3(x)

        x = F.max_unpool2d(
            x, indices=ind2, kernel_size=2, stride=2, output_size=size1)
        x = self.deco4(x)

        x = F.max_unpool2d(x, indices=ind1, kernel_size=2, stride=2)
        x = self.deco5(x)

        logit_list.append(x)

        return logit_list




'''$models/unet'''

@MODELS.add_component
class UNet(nn.Layer):


    def __init__(self,
                 num_classes,
                 align_corners=False,
                 use_deconv=False,
                 pretrained=None):
        super().__init__()

        self.encode = Encoder()
        self.decode = DecoderUNet(align_corners, use_deconv=use_deconv)
        self.cls = self.conv = nn.Conv2D(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        logit_list = []
        x, short_cuts = self.encode(x)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            load_entire_model(self, self.pretrained)


class Encoder(nn.Layer):
    def __init__(self):
        super().__init__()

        self.double_conv = nn.Sequential(
            ConvBNReLU(3, 64, 3), ConvBNReLU(64, 64, 3))
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel[0], channel[1])
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(ConvBNReLU(in_channels, out_channels, 3))
        modules.append(ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class DecoderUNet(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()

        up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1], align_corners, use_deconv)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class UpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_deconv=False):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                in_channels,
                out_channels // 2,
                kernel_size=2,
                stride=2,
                padding=0)
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2

        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3),
            ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                paddle.shape(short_cut)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x




'''$utils/download'''

lasttime = time.time()
FLUSH_INTERVAL = 0.1

def progress(str, end=False):
    global lasttime
    if end:
        str += "\n"
        lasttime = 0
    if time.time() - lasttime >= FLUSH_INTERVAL:
        sys.stdout.write("\r%s" % str)
        lasttime = time.time()
        sys.stdout.flush()

def _download_file(url, savepath, print_progress):
    if print_progress:
        print("Connecting to {}".format(url))
    r = requests.get(url, stream=True, timeout=15)
    total_length = r.headers.get('content-length')

    if total_length is None:
        with open(savepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    else:
        with open(savepath, 'wb') as f:
            dl = 0
            total_length = int(total_length)
            starttime = time.time()
            if print_progress:
                print("Downloading %s" % os.path.basename(savepath))
            for data in r.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                if print_progress:
                    done = int(50 * dl / total_length)
                    progress("[%-50s] %.2f%%" %
                             ('=' * done, float(100 * dl) / total_length))
        if print_progress:
            progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)

def _uncompress_file_zip(filepath, extrapath):
    files = zipfile.ZipFile(filepath, 'r')
    filelist = files.namelist()
    rootpath = filelist[0]
    total_num = len(filelist)
    for index, file in enumerate(filelist):
        files.extract(file, extrapath)
        yield total_num, index, rootpath
    files.close()
    yield total_num, index, rootpath

def _uncompress_file_tar(filepath, extrapath, mode="r:gz"):
    files = tarfile.open(filepath, mode)
    filelist = files.getnames()
    total_num = len(filelist)
    rootpath = filelist[0]
    for index, file in enumerate(filelist):
        files.extract(file, extrapath)
        yield total_num, index, rootpath
    files.close()
    yield total_num, index, rootpath

def _uncompress_file(filepath, extrapath, delete_file, print_progress):
    if print_progress:
        print("Uncompress %s" % os.path.basename(filepath))

    if filepath.endswith("zip"):
        handler = _uncompress_file_zip
    elif filepath.endswith("tgz"):
        handler = functools.partial(_uncompress_file_tar, mode="r:*")
    else:
        handler = functools.partial(_uncompress_file_tar, mode="r")

    for total_num, index, rootpath in handler(filepath, extrapath):
        if print_progress:
            done = int(50 * float(index) / total_num)
            progress(
                "[%-50s] %.2f%%" % ('=' * done, float(100 * index) / total_num))
    if print_progress:
        progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)

    if delete_file:
        os.remove(filepath)

    return rootpath

def download_file_and_uncompress(url,
                                 savepath=None,
                                 extrapath=None,
                                 extraname=None,
                                 print_progress=True,
                                 cover=False,
                                 delete_file=True):
    if savepath is None:
        savepath = "."

    if extrapath is None:
        extrapath = "."

    savename = url.split("/")[-1]
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    savepath = os.path.join(savepath, savename)
    savename = ".".join(savename.split(".")[:-1])
    savename = os.path.join(extrapath, savename)
    extraname = savename if extraname is None else os.path.join(
        extrapath, extraname)

    if cover:
        if os.path.exists(savepath):
            shutil.rmtree(savepath)
        if os.path.exists(savename):
            shutil.rmtree(savename)
        if os.path.exists(extraname):
            shutil.rmtree(extraname)

    if not os.path.exists(extraname):
        if not os.path.exists(savename):
            if not os.path.exists(savepath):
                _download_file(url, savepath, print_progress)

            if (not tarfile.is_tarfile(savepath)) and (
                    not zipfile.is_zipfile(savepath)):
                if not os.path.exists(extraname):
                    os.makedirs(extraname)
                shutil.move(savepath, extraname)
                return extraname

            savename = _uncompress_file(savepath, extrapath, delete_file,
                                        print_progress)
            savename = os.path.join(extrapath, savename)
        shutil.move(savename, extraname)
    return extraname






'''$transforms/transforms'''

@TRANSFORMS.add_component
class Compose:


    def __init__(self, transforms, to_rgb=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb

    def __call__(self, im, label=None):

        if isinstance(im, str):
            im = cv2.imread(im).astype('float32')
        if isinstance(label, str):
            label = np.asarray(PILImage.open(label))
        if im is None:
            raise ValueError('Can\'t read The image file {}!'.format(im))
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
        im = np.transpose(im, (2, 0, 1))
        return (im, label)


@TRANSFORMS.add_component
class RandomHorizontalFlip:


    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im, label=None):
        if random.random() < self.prob:
            im = horizontal_flip(im)
            if label is not None:
                label = horizontal_flip(label)
        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class RandomVerticalFlip:


    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im, label=None):
        if random.random() < self.prob:
            im = vertical_flip(im)
            if label is not None:
                label = vertical_flip(label)
        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class Resize:


    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=(512, 512), interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                        format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                    .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im, label=None):


        if not isinstance(im, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im = resize(im, self.target_size, self.interp_dict[interp])
        if label is not None:
            label = resize(label, self.target_size,
                                      cv2.INTER_NEAREST)

        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class ResizeByLong:
    """
    Resize the long side of an image to given size, and then scale the other side proportionally.

    Args:
        long_size (int): The target size of long side.
    """

    def __init__(self, long_size):
        self.long_size = long_size

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        im = resize_long(im, self.long_size)
        if label is not None:
            label = resize_long(label, self.long_size,
                                           cv2.INTER_NEAREST)

        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class ResizeByShort:
    """
    Resize the short side of an image to given size, and then scale the other side proportionally.

    Args:
        short_size (int): The target size of short side.
    """

    def __init__(self, short_size):
        self.short_size = short_size

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        im = resize_short(im, self.short_size)
        if label is not None:
            label = resize_short(label, self.short_size,
                                            cv2.INTER_NEAREST)

        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class LimitLong:
    """
    Limit the long edge of image.

    If the long edge is larger than max_long, resize the long edge
    to max_long, while scale the short edge proportionally.

    If the long edge is smaller than min_long, resize the long edge
    to min_long, while scale the short edge proportionally.

    Args:
        max_long (int, optional): If the long edge of image is larger than max_long,
            it will be resize to max_long. Default: None.
        min_long (int, optional): If the long edge of image is smaller than min_long,
            it will be resize to min_long. Default: None.
    """

    def __init__(self, max_long=None, min_long=None):
        if max_long is not None:
            if not isinstance(max_long, int):
                raise TypeError(
                    "Type of `max_long` is invalid. It should be int, but it is {}"
                        .format(type(max_long)))
        if min_long is not None:
            if not isinstance(min_long, int):
                raise TypeError(
                    "Type of `min_long` is invalid. It should be int, but it is {}"
                        .format(type(min_long)))
        if (max_long is not None) and (min_long is not None):
            if min_long > max_long:
                raise ValueError(
                    '`max_long should not smaller than min_long, but they are {} and {}'
                        .format(max_long, min_long))
        self.max_long = max_long
        self.min_long = min_long

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """
        h, w = im.shape[0], im.shape[1]
        long_edge = max(h, w)
        target = long_edge
        if (self.max_long is not None) and (long_edge > self.max_long):
            target = self.max_long
        elif (self.min_long is not None) and (long_edge < self.min_long):
            target = self.min_long

        if target != long_edge:
            im = resize_long(im, target)
            if label is not None:
                label = resize_long(label, target, cv2.INTER_NEAREST)

        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class ResizeRangeScaling:
    """
    Resize the long side of an image into a range, and then scale the other side proportionally.

    Args:
        min_value (int, optional): The minimum value of long side after resize. Default: 400.
        max_value (int, optional): The maximum value of long side after resize. Default: 600.
    """

    def __init__(self, min_value=400, max_value=600):
        if min_value > max_value:
            raise ValueError('min_value must be less than max_value, '
                             'but they are {} and {}.'.format(
                min_value, max_value))
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.min_value == self.max_value:
            random_size = self.max_value
        else:
            random_size = int(
                np.random.uniform(self.min_value, self.max_value) + 0.5)
        im = resize_long(im, random_size, cv2.INTER_LINEAR)
        if label is not None:
            label = resize_long(label, random_size,
                                           cv2.INTER_NEAREST)

        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class ResizeStepScaling:
    """
    Scale an image proportionally within a range.

    Args:
        min_scale_factor (float, optional): The minimum scale. Default: 0.75.
        max_scale_factor (float, optional): The maximum scale. Default: 1.25.
        scale_step_size (float, optional): The scale interval. Default: 0.25.

    Raises:
        ValueError: When min_scale_factor is smaller than max_scale_factor.
    """

    def __init__(self,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 scale_step_size=0.25):
        if min_scale_factor > max_scale_factor:
            raise ValueError(
                'min_scale_factor must be less than max_scale_factor, '
                'but they are {} and {}.'.format(min_scale_factor,
                                                 max_scale_factor))
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.min_scale_factor == self.max_scale_factor:
            scale_factor = self.min_scale_factor

        elif self.scale_step_size == 0:
            scale_factor = np.random.uniform(self.min_scale_factor,
                                             self.max_scale_factor)

        else:
            num_steps = int((self.max_scale_factor - self.min_scale_factor) /
                            self.scale_step_size + 1)
            scale_factors = np.linspace(self.min_scale_factor,
                                        self.max_scale_factor,
                                        num_steps).tolist()
            np.random.shuffle(scale_factors)
            scale_factor = scale_factors[0]
        w = int(round(scale_factor * im.shape[1]))
        h = int(round(scale_factor * im.shape[0]))

        im = resize(im, (w, h), cv2.INTER_LINEAR)
        if label is not None:
            label = resize(label, (w, h), cv2.INTER_NEAREST)

        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class Normalize:
    """
    Normalize an image.

    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].

    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std)

        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class Padding:
    """
    Add bottom-right padding to a raw image or annotation image.

    Args:
        target_size (list|tuple): The target size after padding.
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.

    Raises:
        TypeError: When target_size is neither list nor tuple.
        ValueError: When the length of target_size is not 2.
    """

    def __init__(self,
                 target_size,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                        format(target_size))
        else:
            raise TypeError(
                "Type of target_size is invalid. It should be list or tuple, now is {}"
                    .format(type(target_size)))
        self.target_size = target_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        im_height, im_width = im.shape[0], im.shape[1]
        if isinstance(self.target_size, int):
            target_height = self.target_size
            target_width = self.target_size
        else:
            target_height = self.target_size[1]
            target_width = self.target_size[0]
        pad_height = target_height - im_height
        pad_width = target_width - im_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                'The size of image should be less than `target_size`, but the size of image ({}, {}) is larger than `target_size` ({}, {})'
                    .format(im_width, im_height, target_width, target_height))
        else:
            im = cv2.copyMakeBorder(
                im,
                0,
                pad_height,
                0,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=self.im_padding_value)
            if label is not None:
                label = cv2.copyMakeBorder(
                    label,
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=self.label_padding_value)
        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class PaddingByAspectRatio:
    """

    Args:
        aspect_ratio (int|float, optional): The aspect ratio = width / height. Default: 1.
    """

    def __init__(self,
                 aspect_ratio=1,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        self.aspect_ratio = aspect_ratio
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        img_height = im.shape[0]
        img_width = im.shape[1]
        ratio = img_width / img_height
        if ratio == self.aspect_ratio:
            if label is None:
                return (im,)
            else:
                return (im, label)
        elif ratio > self.aspect_ratio:
            img_height = int(img_width / self.aspect_ratio)
        else:
            img_width = int(img_height * self.aspect_ratio)
        padding = Padding((img_width, img_height),
                          im_padding_value=self.im_padding_value,
                          label_padding_value=self.label_padding_value)
        return padding(im, label)


@TRANSFORMS.add_component
class RandomPaddingCrop:
    """
    Crop a sub-image from a raw image and annotation image randomly. If the target cropping size
    is larger than original image, then the bottom-right padding will be added.

    Args:
        crop_size (tuple, optional): The target cropping size. Default: (512, 512).
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.

    Raises:
        TypeError: When crop_size is neither list nor tuple.
        ValueError: When the length of crop_size is not 2.
    """

    def __init__(self,
                 crop_size=(512, 512),
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 2:
                raise ValueError(
                    'Type of `crop_size` is list or tuple. It should include 2 elements, but it is {}'
                        .format(crop_size))
        else:
            raise TypeError(
                "The type of `crop_size` is invalid. It should be list or tuple, but it is {}"
                    .format(type(crop_size)))
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if isinstance(self.crop_size, int):
            crop_width = self.crop_size
            crop_height = self.crop_size
        else:
            crop_width = self.crop_size[0]
            crop_height = self.crop_size[1]

        img_height = im.shape[0]
        img_width = im.shape[1]

        if img_height == crop_height and img_width == crop_width:
            if label is None:
                return (im,)
            else:
                return (im, label)
        else:
            pad_height = max(crop_height - img_height, 0)
            pad_width = max(crop_width - img_width, 0)
            if (pad_height > 0 or pad_width > 0):
                im = cv2.copyMakeBorder(
                    im,
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=self.im_padding_value)
                if label is not None:
                    label = cv2.copyMakeBorder(
                        label,
                        0,
                        pad_height,
                        0,
                        pad_width,
                        cv2.BORDER_CONSTANT,
                        value=self.label_padding_value)
                img_height = im.shape[0]
                img_width = im.shape[1]

            if crop_height > 0 and crop_width > 0:
                h_off = np.random.randint(img_height - crop_height + 1)
                w_off = np.random.randint(img_width - crop_width + 1)

                im = im[h_off:(crop_height + h_off), w_off:(
                        w_off + crop_width), :]
                if label is not None:
                    label = label[h_off:(crop_height + h_off), w_off:(
                            w_off + crop_width)]
        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class RandomCenterCrop:
    """
    Crops the given the input data at the center.
    Args:
        retain_ratio (tuple or list, optional): The length of the input list or tuple must be 2. Default: (0.5, 0.5).
        the first value is used for width and the second is for height.
        In addition, the minimum size of the cropped image is [width * retain_ratio[0], height * retain_ratio[1]].
    Raises:
        TypeError: When retain_ratio is neither list nor tuple. Default: None.
        ValueError: When the value of retain_ratio is not in [0-1].
    """

    def __init__(self,
                 retain_ratio=(0.5, 0.5)):
        if isinstance(retain_ratio, list) or isinstance(retain_ratio, tuple):
            if len(retain_ratio) != 2:
                raise ValueError(
                    'When type of `retain_ratio` is list or tuple, it shoule include 2 elements, but it is {}'.format(
                        retain_ratio)
                )
            if retain_ratio[0] > 1 or retain_ratio[1] > 1 or retain_ratio[0] < 0 or retain_ratio[1] < 0:
                raise ValueError(
                    'Value of `retain_ratio` should be in [0, 1], but it is {}'.format(retain_ratio)
                )
        else:
            raise TypeError(
                "The type of `retain_ratio` is invalid. It should be list or tuple, but it is {}"
                    .format(type(retain_ratio)))
        self.retain_ratio = retain_ratio

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.
        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """
        retain_width = self.retain_ratio[0]
        retain_height = self.retain_ratio[1]

        img_height = im.shape[0]
        img_width = im.shape[1]

        if retain_width == 1. and retain_height == 1.:
            if label is None:
                return (im,)
            else:
                return (im, label)
        else:
            randw = np.random.randint(img_width * (1 - retain_width))
            randh = np.random.randint(img_height * (1 - retain_height))
            offsetw = 0 if randw == 0 else np.random.randint(randw)
            offseth = 0 if randh == 0 else np.random.randint(randh)
            p0, p1, p2, p3 = offseth, img_height + offseth - randh, offsetw, img_width + offsetw - randw
            im = im[p0:p1, p2:p3, :]
            if label is not None:
                label = label[p0:p1, p2:p3, :]

        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class ScalePadding:
    """
        Add center padding to a raw image or annotation image,then scale the
        image to target size.

        Args:
            target_size (list|tuple, optional): The target size of image. Default: (512, 512).
            im_padding_value (list, optional): The padding value of raw image.
                Default: [127.5, 127.5, 127.5].
            label_padding_value (int, optional): The padding value of annotation image. Default: 255.

        Raises:
            TypeError: When target_size is neither list nor tuple.
            ValueError: When the length of target_size is not 2.
    """

    def __init__(self,
                 target_size=(512, 512),
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                        format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                    .format(type(target_size)))

        self.target_size = target_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """
        height = im.shape[0]
        width = im.shape[1]

        new_im = np.zeros(
            (max(height, width), max(height, width), 3)) + self.im_padding_value
        if label is not None:
            new_label = np.zeros((max(height, width), max(
                height, width))) + self.label_padding_value

        if height > width:
            padding = int((height - width) / 2)
            new_im[:, padding:padding + width, :] = im
            if label is not None:
                new_label[:, padding:padding + width] = label
        else:
            padding = int((width - height) / 2)
            new_im[padding:padding + height, :, :] = im
            if label is not None:
                new_label[padding:padding + height, :] = label

        im = np.uint8(new_im)
        im = resize(im, self.target_size, interp=cv2.INTER_CUBIC)
        if label is not None:
            label = np.uint8(new_label)
            label = resize(
                label, self.target_size, interp=cv2.INTER_CUBIC)
        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class RandomNoise:
    """
    Superimposing noise on an image with a certain probability.

    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.5.
        max_sigma(float, optional): The maximum value of standard deviation of the distribution.
            Default: 10.0.
    """

    def __init__(self, prob=0.5, max_sigma=10.0):
        self.prob = prob
        self.max_sigma = max_sigma

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """
        if random.random() < self.prob:
            mu = 0
            sigma = random.random() * self.max_sigma
            im = np.array(im, dtype=np.float32)
            im += np.random.normal(mu, sigma, im.shape)
            im[im > 255] = 255
            im[im < 0] = 0

        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class RandomBlur:
    """
    Blurring an image by a Gaussian function with a certain probability.

    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.1.
        blur_type(str, optional): A type of blurring an image,
            gaussian stands for cv2.GaussianBlur,
            median stands for cv2.medianBlur,
            blur stands for cv2.blur,
            random represents randomly selected from above.
            Default: gaussian.
    """

    def __init__(self, prob=0.1, blur_type="gaussian"):
        self.prob = prob
        self.blur_type = blur_type

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                im = np.array(im, dtype='uint8')
                if self.blur_type == "gaussian":
                    im = cv2.GaussianBlur(im, (radius, radius), 0, 0)
                elif self.blur_type == "median":
                    im = cv2.medianBlur(im, radius)
                elif self.blur_type == "blur":
                    im = cv2.blur(im, (radius, radius))
                elif self.blur_type == "random":
                    select = random.random()
                    if select < 0.3:
                        im = cv2.GaussianBlur(im, (radius, radius), 0)
                    elif select < 0.6:
                        im = cv2.medianBlur(im, radius)
                    else:
                        im = cv2.blur(im, (radius, radius))
                else:
                    im = cv2.GaussianBlur(im, (radius, radius), 0, 0)
        im = np.array(im, dtype='float32')
        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class RandomRotation:
    """
    Rotate an image randomly with padding.

    Args:
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.
    """

    def __init__(self,
                 max_rotation=15,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        self.max_rotation = max_rotation
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.max_rotation > 0:
            (h, w) = im.shape[:2]
            do_rotation = np.random.uniform(-self.max_rotation,
                                            self.max_rotation)
            pc = (w // 2, h // 2)
            r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
            cos = np.abs(r[0, 0])
            sin = np.abs(r[0, 1])

            nw = int((h * sin) + (w * cos))
            nh = int((h * cos) + (w * sin))

            (cx, cy) = pc
            r[0, 2] += (nw / 2) - cx
            r[1, 2] += (nh / 2) - cy
            dsize = (nw, nh)
            im = cv2.warpAffine(
                im,
                r,
                dsize=dsize,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.im_padding_value)
            if label is not None:
                label = cv2.warpAffine(
                    label,
                    r,
                    dsize=dsize,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=self.label_padding_value)

        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class RandomScaleAspect:
    """
    Crop a sub-image from an original image with a range of area ratio and aspect and
    then scale the sub-image back to the size of the original image.

    Args:
        min_scale (float, optional): The minimum area ratio of cropped image to the original image. Default: 0.5.
        aspect_ratio (float, optional): The minimum aspect ratio. Default: 0.33.
    """

    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height = im.shape[0]
            img_width = im.shape[1]
            for i in range(0, 10):
                area = img_height * img_width
                target_area = area * np.random.uniform(self.min_scale, 1.0)
                aspectRatio = np.random.uniform(self.aspect_ratio,
                                                1.0 / self.aspect_ratio)

                dw = int(np.sqrt(target_area * 1.0 * aspectRatio))
                dh = int(np.sqrt(target_area * 1.0 / aspectRatio))
                if (np.random.randint(10) < 5):
                    tmp = dw
                    dw = dh
                    dh = tmp

                if (dh < img_height and dw < img_width):
                    h1 = np.random.randint(0, img_height - dh)
                    w1 = np.random.randint(0, img_width - dw)

                    im = im[h1:(h1 + dh), w1:(w1 + dw), :]
                    im = cv2.resize(
                        im, (img_width, img_height),
                        interpolation=cv2.INTER_LINEAR)
                    if label is not None:
                        label = label[h1:(h1 + dh), w1:(w1 + dw)]
                        label = cv2.resize(
                            label, (img_width, img_height),
                            interpolation=cv2.INTER_NEAREST)
                    break
        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class RandomDistort:

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5,
                 sharpness_range=0.5,
                 sharpness_prob=0):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob
        self.sharpness_range = sharpness_range
        self.sharpness_prob = sharpness_prob

    def __call__(self, im, label=None):

        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        sharpness_lower = 1 - self.sharpness_range
        sharpness_upper = 1 + self.sharpness_range
        ops = [
            brightness, contrast, saturation,
            hue, sharpness
        ]
        random.shuffle(ops)
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            },
            'sharpness': {
                'sharpness_lower': sharpness_lower,
                'sharpness_upper': sharpness_upper,
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob,
            'sharpness': self.sharpness_prob
        }
        im = im.astype('uint8')
        im = PILImage.fromarray(im)
        for id in range(len(ops)):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['im'] = im
            if np.random.uniform(0, 1) < prob:
                im = ops[id](**params)
        im = np.asarray(im).astype('float32')
        if label is None:
            return (im,)
        else:
            return (im, label)


@TRANSFORMS.add_component
class RandomAffine:

    def __init__(self,
                 size=(224, 224),
                 translation_offset=0,
                 max_rotation=15,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 im_padding_value=(128, 128, 128),
                 label_padding_value=(255, 255, 255)):
        self.size = size
        self.translation_offset = translation_offset
        self.max_rotation = max_rotation
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, label=None):

        w, h = self.size
        bbox = [0, 0, im.shape[1] - 1, im.shape[0] - 1]
        x_offset = (random.random() - 0.5) * 2 * self.translation_offset
        y_offset = (random.random() - 0.5) * 2 * self.translation_offset
        dx = (w - (bbox[2] + bbox[0])) / 2.0
        dy = (h - (bbox[3] + bbox[1])) / 2.0

        matrix_trans = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])

        angle = random.random() * 2 * self.max_rotation - self.max_rotation
        scale = random.random() * (self.max_scale_factor - self.min_scale_factor
                                   ) + self.min_scale_factor
        scale *= np.mean(
            [float(w) / (bbox[2] - bbox[0]),
             float(h) / (bbox[3] - bbox[1])])
        alpha = scale * math.cos(angle / 180.0 * math.pi)
        beta = scale * math.sin(angle / 180.0 * math.pi)

        centerx = w / 2.0 + x_offset
        centery = h / 2.0 + y_offset
        matrix = np.array(
            [[alpha, beta, (1 - alpha) * centerx - beta * centery],
             [-beta, alpha, beta * centerx + (1 - alpha) * centery],
             [0, 0, 1.0]])

        matrix = matrix.dot(matrix_trans)[0:2, :]
        im = cv2.warpAffine(
            np.uint8(im),
            matrix,
            tuple(self.size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.im_padding_value)
        if label is not None:
            label = cv2.warpAffine(
                np.uint8(label),
                matrix,
                tuple(self.size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT)
        if label is None:
            return (im,)
        else:
            return (im, label)




'''probar'''
class Progbar(object):
    """
    Displays a progress bar.
        It refers to https://github.com/keras-team/keras/blob/keras-2/keras/utils/generic_utils.py

    Args:
        target (int): Total number of steps expected, None if unknown.
        width (int): Progress bar width on screen.
        verbose (int): Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics (list|tuple): Iterable of string names of metrics that should *not* be
            averaged over time. Metrics in this list will be displayed as-is. All
            others will be averaged by the progbar before display.
        interval (float): Minimum visual progress update interval (in seconds).
        unit_name (str): Display name for step counts (usually "step" or "sample").
    """

    def __init__(self,
                 target,
                 width=30,
                 verbose=1,
                 interval=0.05,
                 stateful_metrics=None,
                 unit_name='step'):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stderr, 'isatty')
                                  and sys.stderr.isatty())
                                 or 'ipykernel' in sys.modules
                                 or 'posix' in sys.modules
                                 or 'PYCHARM_HOSTED' in os.environ)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None, finalize=None):
        """
        Updates the progress bar.

        Args:
            current (int): Index of current step.
            values (list): List of tuples: `(name, value_for_last_step)`. If `name` is in
                `stateful_metrics`, `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
            finalize (bool): Whether this is the last update for the progress bar. If
                `None`, defaults to `current >= self.target`.
        """

        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target

        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                # In the case that progress bar doesn't have a target value in the first
                # epoch, both on_batch_end and on_epoch_end will be called, which will
                # cause 'current' and 'self._seen_so_far' to have the same value. Force
                # the minimal value to 1 here, otherwise stateful_metric will be 0s.
                value_base = max(current - self._seen_so_far, 1)
                if k not in self._values:
                    self._values[k] = [v * value_base, value_base]
                else:
                    self._values[k][0] += v * value_base
                    self._values[k][1] += value_base
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if now - self._last_update < self.interval and not finalize:
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stderr.write('\b' * prev_total_width)
                sys.stderr.write('\r')
            else:
                sys.stderr.write('\n')

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stderr.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0

            if self.target is None or finalize:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)
            else:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if finalize:
                info += '\n'

            sys.stderr.write(info)
            sys.stderr.flush()

        elif self.verbose == 2:
            if finalize:
                numdigits = int(np.log10(self.target)) + 1
                count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stderr.write(info)
                sys.stderr.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)






'''$utils/visualize'''
def visualize(image, result, color_map, save_dir=None, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c3, c2, c1))

    im = cv2.imread(image)
    vis_result = cv2.addWeighted(im, 1, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, vis_result)
    else:
        return vis_result

def get_pseudo_color_map(pred, color_map=None):
    """
    Get the pseudo color image.

    Args:
        pred (numpy.ndarray): the origin predicted image.
        color_map (list, optional): the palette color map. Default: None,
            use paddleseg's default color map.
    
    Returns:
        (numpy.ndarray): the pseduo image.
    """
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask

def get_color_map_list(num_classes, custom_color=None):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map






'''core目录infer'''
def get_reverse_list(ori_shape, transforms):
    """
    get reverse list of transform.

    Args:
        ori_shape (list): Origin shape of image.
        transforms (list): List of transform.

    Returns:
        list: List of tuple, there are two format:
            ('resize', (h, w)) The image shape before resize,
            ('padding', (h, w)) The image shape before padding.
    """
    reverse_list = []
    h, w = ori_shape[0], ori_shape[1]
    for op in transforms:
        if op.__class__.__name__ in ['Resize']:
            reverse_list.append(('resize', (h, w)))
            h, w = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['ResizeByLong']:
            reverse_list.append(('resize', (h, w)))
            long_edge = max(h, w)
            short_edge = min(h, w)
            short_edge = int(round(short_edge * op.long_size / long_edge))
            long_edge = op.long_size
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
        if op.__class__.__name__ in ['ResizeByShort']:
            reverse_list.append(('resize', (h, w)))
            long_edge = max(h, w)
            short_edge = min(h, w)
            long_edge = int(round(long_edge * op.short_size / short_edge))
            short_edge = op.short_size
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
        if op.__class__.__name__ in ['Padding']:
            reverse_list.append(('padding', (h, w)))
            w, h = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['PaddingByAspectRatio']:
            reverse_list.append(('padding', (h, w)))
            ratio = w / h
            if ratio == op.aspect_ratio:
                pass
            elif ratio > op.aspect_ratio:
                h = int(w / op.aspect_ratio)
            else:
                w = int(h * op.aspect_ratio)
        if op.__class__.__name__ in ['LimitLong']:
            long_edge = max(h, w)
            short_edge = min(h, w)
            if ((op.max_long is not None) and (long_edge > op.max_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.max_long
                short_edge = int(round(short_edge * op.max_long / long_edge))
            elif ((op.min_long is not None) and (long_edge < op.min_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.min_long
                short_edge = int(round(short_edge * op.min_long / long_edge))
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
    return reverse_list

def reverse_transform(pred, ori_shape, transforms, mode='nearest'):
    """recover pred to origin shape"""
    reverse_list = get_reverse_list(ori_shape, transforms)
    intTypeList = [paddle.int8, paddle.int16, paddle.int32, paddle.int64]
    dtype = pred.dtype
    for item in reverse_list[::-1]:
        if item[0] == 'resize':
            h, w = item[1][0], item[1][1]
            if paddle.get_device() == 'cpu' and dtype in intTypeList:
                pred = paddle.cast(pred, 'float32')
                pred = F.interpolate(pred, (h, w), mode=mode)
                pred = paddle.cast(pred, dtype)
            else:
                pred = F.interpolate(pred, (h, w), mode=mode)
        elif item[0] == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred

def flip_combination(flip_horizontal=False, flip_vertical=False):
    """
    Get flip combination.

    Args:
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.

    Returns:
        list: List of tuple. The first element of tuple is whether to flip horizontally,
            and the second is whether to flip vertically.
    """

    flip_comb = [(False, False)]
    if flip_horizontal:
        flip_comb.append((True, False))
    if flip_vertical:
        flip_comb.append((False, True))
        if flip_horizontal:
            flip_comb.append((True, True))
    return flip_comb

def tensor_flip(x, flip):
    """Flip tensor according directions"""
    if flip[0]:
        x = x[:, :, :, ::-1]
    if flip[1]:
        x = x[:, :, ::-1, :]
    return x

def slide_inference(model, im, crop_size, stride):
    """
    Infer by sliding window.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        crop_size (tuple|list). The size of sliding window, (w, h).
        stride (tuple|list). The size of stride, (w, h).

    Return:
        Tensor: The logit of input image.
    """
    h_im, w_im = im.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    # calculate the crop nums
    rows = np.int(np.ceil(1.0 * (h_im - h_crop) / h_stride)) + 1
    cols = np.int(np.ceil(1.0 * (w_im - w_crop) / w_stride)) + 1
    # prevent negative sliding rounds when imgs after scaling << crop_size
    rows = 1 if h_im <= h_crop else rows
    cols = 1 if w_im <= w_crop else cols
    # TODO 'Tensor' object does not support item assignment. If support, use tensor to calculation.
    final_logit = None
    count = np.zeros([1, 1, h_im, w_im])
    for r in range(rows):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop = im[:, :, h1:h2, w1:w2]
            logits = model(im_crop)
            if not isinstance(logits, collections.abc.Sequence):
                raise TypeError(
                    "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                    .format(type(logits)))
            logit = logits[0].numpy()
            if final_logit is None:
                final_logit = np.zeros([1, logit.shape[1], h_im, w_im])
            final_logit[:, :, h1:h2, w1:w2] += logit[:, :, :h2 - h1, :w2 - w1]
            count[:, :, h1:h2, w1:w2] += 1
    if np.sum(count == 0) != 0:
        raise RuntimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    final_logit = final_logit / count
    final_logit = paddle.to_tensor(final_logit)
    return final_logit

def inference(model,
              im,
              ori_shape=None,
              transforms=None,
              is_slide=False,
              stride=None,
              crop_size=None):
    """
    Inference for image.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        ori_shape (list): Origin shape of image.
        transforms (list): Transforms for image.
        is_slide (bool): Whether to infer by sliding window. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.

    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, h, w) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, h, w) is returned.
    """
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        im = im.transpose((0, 2, 3, 1))
    if not is_slide:
        logits = model(im)
        if not isinstance(logits, collections.abc.Sequence):
            raise TypeError(
                "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                .format(type(logits)))
        logit = logits[0]
    else:
        logit = slide_inference(model, im, crop_size=crop_size, stride=stride)
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        logit = logit.transpose((0, 3, 1, 2))
    if ori_shape is not None:
        logit = reverse_transform(logit, ori_shape, transforms, mode='bilinear')
        pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
        return pred, logit
    else:
        return logit

def aug_inference(model,
                  im,
                  ori_shape,
                  transforms,
                  scales=1.0,
                  flip_horizontal=False,
                  flip_vertical=False,
                  is_slide=False,
                  stride=None,
                  crop_size=None):
    """
    Infer with augmentation.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        ori_shape (list): Origin shape of image.
        transforms (list): Transforms for image.
        scales (float|tuple|list):  Scales for resize. Default: 1.
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.
        is_slide (bool): Whether to infer by sliding wimdow. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.

    Returns:
        Tensor: Prediction of image with shape (1, 1, h, w) is returned.
    """
    if isinstance(scales, float):
        scales = [scales]
    elif not isinstance(scales, (tuple, list)):
        raise TypeError(
            '`scales` expects float/tuple/list type, but received {}'.format(
                type(scales)))
    final_logit = 0
    h_input, w_input = im.shape[-2], im.shape[-1]
    flip_comb = flip_combination(flip_horizontal, flip_vertical)
    for scale in scales:
        h = int(h_input * scale + 0.5)
        w = int(w_input * scale + 0.5)
        im = F.interpolate(im, (h, w), mode='bilinear')
        for flip in flip_comb:
            im_flip = tensor_flip(im, flip)
            logit = inference(
                model,
                im_flip,
                is_slide=is_slide,
                crop_size=crop_size,
                stride=stride)
            logit = tensor_flip(logit, flip)
            logit = F.interpolate(logit, (h_input, w_input), mode='bilinear')

            logit = F.softmax(logit, axis=1)
            final_logit = final_logit + logit

    final_logit = reverse_transform(
        final_logit, ori_shape, transforms, mode='bilinear')
    pred = paddle.argmax(final_logit, axis=1, keepdim=True, dtype='int32')

    return pred, final_logit



'''根目录predict'''
def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
        else:
            image_dir = os.path.dirname(image_path)
            with open(image_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) > 1:
                        line = line.split()[0]
                    image_list.append(os.path.join(image_dir, line))
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be a path of image, or a file list containing image paths, or a directory including images.'
        )

    if len(image_list) == 0:
        raise RuntimeError(
            'There are not image file in `--image_path`={}'.format(image_path))

    return image_list, image_dir

def get_test_config(cfg, args):

    test_config = cfg.test_config
    if args.aug_pred:
        test_config['aug_pred'] = args.aug_pred
        test_config['scales'] = args.scales

    if args.flip_horizontal:
        test_config['flip_horizontal'] = args.flip_horizontal

    if args.flip_vertical:
        test_config['flip_vertical'] = args.flip_vertical

    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride

    if args.custom_color:
        test_config['custom_color'] = args.custom_color

    return test_config

def config_check(cfg, train_dataset=None, val_dataset=None):
    """
    To check config。

    Args:
        cfg (paddleseg.cvlibs.Config): An object of paddleseg.cvlibs.Config.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
    """

    num_classes_check(cfg, train_dataset, val_dataset)


def num_classes_check(cfg, train_dataset, val_dataset):
    """"
    Check that the num_classes in model, train_dataset and val_dataset is consistent.
    """
    num_classes_set = set()
    if train_dataset and hasattr(train_dataset, 'num_classes'):
        num_classes_set.add(train_dataset.num_classes)
    if val_dataset and hasattr(val_dataset, 'num_classes'):
        num_classes_set.add(val_dataset.num_classes)
    if cfg.dic.get('model', None) and cfg.dic['model'].get('num_classes', None):
        num_classes_set.add(cfg.dic['model'].get('num_classes'))
    if (not cfg.train_dataset) and (not cfg.val_dataset):
        raise ValueError(
            'One of `train_dataset` or `val_dataset should be given, but there are none.'
        )
    if len(num_classes_set) == 0:
        raise ValueError(
            '`num_classes` is not found. Please set it in model, train_dataset or val_dataset'
        )
    elif len(num_classes_set) > 1:
        raise ValueError(
            '`num_classes` is not consistent: {}. Please set it consistently in model or train_dataset or val_dataset'
            .format(num_classes_set))
    else:
        num_classes = num_classes_set.pop()
        if train_dataset:
            train_dataset.num_classes = num_classes
        if val_dataset:
            val_dataset.num_classes = num_classes

def predict1(cfg, model_path, save_dir, image_path, custom_color):
    
    args = parse_args()
    args.cfg = cfg
    args.model_path = model_path
    args.save_dir = save_dir
    args.image_path = image_path
    args.aug_pred = False
    args.scales = 1.0
    args.flip_horizontal = False
    args.flip_vertical = False
    args.is_slide = False
    args.crop_size = None
    args.stride = None
    args.custom_color = custom_color
    args.parse_args()
    
    
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    
    val_dataset = cfg.val_dataset
    if not val_dataset:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )
    
    
    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    #info(msg)

    model = cfg.model
    transforms = val_dataset.transforms
    image_list, image_dir = get_image_list(args.image_path)
    info('Number of predict images = {}'.format(len(image_list)))

    test_config = get_test_config(cfg, args)
    #config_check(cfg, val_dataset=val_dataset)

    c_pixel, percent = predict2(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        **test_config)
    return c_pixel, percent


'''core目录predict'''

def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


@contextlib.contextmanager
def generate_tempdir(directory: str = None, **kwargs):
    '''Generate a temporary directory'''
    directory = TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir


def load_entire_model(model, pretrained):
    if pretrained is not None:
        load_pretrained_model(model, pretrained)
    else:
        warning('Not all pretrained params of {} are loaded, ' \
                       'training from scratch or a pretrained backbone.'.format(model.__class__.__name__))


def download_pretrained_model(pretrained_model):
    """
    Download pretrained model from url.
    Args:
        pretrained_model (str): the url of pretrained weight
    Returns:
        str: the path of pretrained weight
    """
    assert urlparse(pretrained_model).netloc, "The url is not valid."

    pretrained_model = unquote(pretrained_model)
    savename = pretrained_model.split('/')[-1]
    if not savename.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
        savename = pretrained_model.split('/')[-2]
    else:
        savename = savename.split('.')[0]

    with generate_tempdir() as _dir:
        with filelock.FileLock(os.path.join(TMP_HOME, savename)):
            pretrained_model = download_file_and_uncompress(
                pretrained_model,
                savepath=_dir,
                extrapath=PRETRAINED_MODEL_HOME,
                extraname=savename)
            pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
    return pretrained_model

def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        info('Loading pretrained model from {}'.format(pretrained_model))

        if urlparse(pretrained_model).netloc:
            pretrained_model = download_pretrained_model(pretrained_model)
            
        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(
                        model_state_dict[k].shape):
                    warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape,
                                model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict),
                model.__class__.__name__))

        else:
            raise ValueError(
                'The pretrained model directory is not Found: {}'.format(
                    pretrained_model))
    else:
        info(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))

def predict2(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None,
            custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')

    info("Start to predict...")
    progbar_pred = Progbar(target=len(img_lists[0]), verbose=1)
    color_map = get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            im = cv2.imread(im_path)
            ori_shape = im.shape[:2]
            im, _ = transforms(im)
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            if aug_pred:
                pred, _  = aug_inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred, _ = inference(
                    model,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            # get the saved name
            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # save added image
            added_image = visualize(
                im_path, pred, color_map, weight=0.6)
            added_image_path = os.path.join(added_saved_dir, im_file)
            mkdir(added_image_path)
            cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            
            pred_mask = get_pseudo_color_map(pred, color_map)
            pred_saved_path = os.path.join(
                pred_saved_dir,
                os.path.splitext(im_file)[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            # pred_im = utils.visualize(im_path, pred, weight=0.0)
            # pred_saved_path = os.path.join(pred_saved_dir, im_file)
            # mkdir(pred_saved_path)
            # cv2.imwrite(pred_saved_path, pred_im)

            progbar_pred.update(i + 1)
    return Counter(pred.flatten())[1], np.mean(pred)




class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        
        screenRect = QApplication.desktop().screenGeometry()
        height = screenRect.height()
        width = screenRect.width()


        self.setWindowTitle("船舶腐蚀检测与评估系统")
        icon = QIcon()
        icon.addPixmap(QPixmap("C:/Users/11267/Desktop/1.webp"), QIcon.Normal, QIcon.Off)
        self.setWindowIcon(icon)

        grid_layout = QGridLayout()
        main_layout = QVBoxLayout()
        top_bar_layout = QHBoxLayout()
        color_layout = QVBoxLayout()
        back_rgb_layout = QVBoxLayout()
        fore_rgb_layout = QVBoxLayout()
        image_bar_layout = QHBoxLayout()
        
        selection_layout = QVBoxLayout()
        status1 = QHBoxLayout()
        status2 = QHBoxLayout()
        status_layout = QVBoxLayout()
        
        
        self.source_filename = None
        self.source_image_data = None
        self.result_image_data = None
        self.max_img_height = 1400
        self.max_img_width = 1600
        self.model_name = None


        self.select_model_type = QComboBox()
        self.select_model_type.addItems(["点击选择模型类型",
                                         "FastFCN",
                                         "DeepLabV3 (ResNet-50)",
                                         "DeepLabV3P (ResNet-50)",
                                         "PP-LiteSeg",
                                         "FCN (HRNet-W18)",
                                         "FCN (HRNet-W48)",
                                         "SegFormer",
                                         "PSPNet(ResNet-101)"])
        self.select_model_type.setFixedHeight(80)
        self.select_model_type.currentTextChanged.connect(self.choose_model_type)



        self.status_label = QLabel("检测状态:")
        self.status_label.setFont(QFont('宋体', 16))
        self.status_label.setMinimumWidth(int(width/4))
        self.status_label.setAlignment(Qt.AlignCenter)
        
        select_model_button = QPushButton("选择模型文件")        
        select_image_button = QPushButton('选择图像')
        process_image_button = QPushButton('开始检测')
        select_model_button.clicked.connect(self.choose_model)
        select_image_button.clicked.connect(self.choose_source_image)
        process_image_button.clicked.connect(self.process_image)
        
        
        for btn in [select_model_button, select_image_button, process_image_button]:
            btn.setFixedHeight(100)
            #btn.setFixedWidth(300)
        
        self.fore_r_select = QSpinBox()
        self.fore_g_select = QSpinBox()
        self.fore_b_select = QSpinBox()
        
        self.back_r_select = QSpinBox()
        self.back_g_select = QSpinBox()
        self.back_b_select = QSpinBox()
        
        
        for start_val_1, prefix_1, spinbox_1 in zip([0, 255, 0], ['R', 'G', 'B'],
                                              [self.fore_r_select, self.fore_g_select, self.fore_b_select]):
            spinbox_1.setPrefix(f'{prefix_1}:')
            spinbox_1.setMinimum(0)
            spinbox_1.setMaximum(255)
            spinbox_1.setValue(start_val_1)
            spinbox_1.valueChanged.connect(self.choose_params)

        for start_val_2, prefix_2, spinbox_2 in zip([0, 0, 0], ['R', 'G', 'B'],
                                              [self.back_r_select, self.back_g_select, self.back_b_select]):
            spinbox_2.setPrefix(f'{prefix_2}:')
            spinbox_2.setMinimum(0)
            spinbox_2.setMaximum(255)
            spinbox_2.setValue(start_val_2)
            spinbox_2.valueChanged.connect(self.choose_params)

        top_bar_layout.addWidget(self.status_label)
        top_bar_layout.addWidget(self.select_model_type)
        top_bar_layout.addWidget(select_model_button)
        top_bar_layout.addWidget(select_image_button)
        top_bar_layout.addWidget(process_image_button)
        
        fore_rgb_layout.addWidget(QLabel("前景RGB:"))
        fore_rgb_layout.addWidget(self.fore_r_select)
        fore_rgb_layout.addWidget(self.fore_g_select)
        fore_rgb_layout.addWidget(self.fore_b_select)
        
        back_rgb_layout.addWidget(QLabel("背景RGB:"))
        back_rgb_layout.addWidget(self.back_r_select)
        back_rgb_layout.addWidget(self.back_g_select)
        back_rgb_layout.addWidget(self.back_b_select)
        
        
        self.source_image = ImageWidget()
        self.result_image = ImageWidget()
        self.source_image.setMaximumSize(self.max_img_width, self.max_img_height)
        self.result_image.setMaximumSize(self.max_img_width, self.max_img_height)

        source_image_layout = QVBoxLayout()
        original_image = QLabel("原图像:")
        original_image.setMinimumWidth(int(width/4))
        original_image.setFont(QFont('宋体', 14))
        source_image_layout.addWidget(original_image)
        source_image_layout.addWidget(self.source_image)

        result_image_layout = QVBoxLayout()
        seg_result = QLabel("分割结果:")
        seg_result.setMinimumWidth(int(width/4))
        seg_result.setFont(QFont('宋体', 14))
        result_image_layout.addWidget(seg_result)
        result_image_layout.addWidget(self.result_image)

        image_bar_layout.addLayout(source_image_layout)
        image_bar_layout.addLayout(result_image_layout)

        bottom_bar_layout = QHBoxLayout()
        self.save_button = QPushButton('保存图像')
        self.save_button.clicked.connect(self.save_as_file)        
        self.save_button.setFixedHeight(100)
        
        self.model_type_label = QLabel("模型类型:")
        self.model_type_label.setFont(QFont('宋体', 16))
        self.model_type_label.setMinimumWidth(int(width/4))
        self.model_type_label.setAlignment(Qt.AlignCenter)
        
        self.model_name_label = QLabel("模型文件:")
        self.model_name_label.setFont(QFont('宋体', 16))
        self.model_name_label.setMinimumWidth(int(width/4))
        self.model_name_label.setAlignment(Qt.AlignCenter)
        
        self.percent_label = QLabel("腐蚀占比:")
        self.percent_label.setFont(QFont('宋体', 16))
        self.percent_label.setMinimumWidth(int(width/4))
        self.percent_label.setAlignment(Qt.AlignCenter)
        
        self.corrosion_pixels = QLabel("腐蚀面积:")
        self.corrosion_pixels.setFont(QFont('宋体', 16))
        self.corrosion_pixels.setMinimumWidth(int(width/4))
        self.corrosion_pixels.setAlignment(Qt.AlignCenter)
        
        self.image_size = QLabel("图像面积:")
        self.image_size.setFont(QFont('宋体', 16))
        self.image_size.setMinimumWidth(int(width/4))
        self.image_size.setAlignment(Qt.AlignCenter)
        
        bottom_bar_layout.addWidget(self.save_button)
        bottom_bar_layout.addWidget(self.model_type_label)
        bottom_bar_layout.addWidget(self.model_name_label)
        bottom_bar_layout.addWidget(self.image_size)
        bottom_bar_layout.addWidget(self.corrosion_pixels)
        bottom_bar_layout.addWidget(self.percent_label)
        
        
        
        
        #新布局设计

        selection_layout.addWidget(self.select_model_type)
        selection_layout.addWidget(select_model_button)
        selection_layout.addWidget(select_image_button)
        selection_layout.addWidget(process_image_button)
        selection_layout.addWidget(self.save_button)
        
        color_layout.addLayout(fore_rgb_layout)
        color_layout.addLayout(back_rgb_layout)
        
        status1.addWidget(self.status_label)
        status1.addWidget(self.model_type_label)
        status1.addWidget(self.model_name_label)
        status2.addWidget(self.image_size)
        status2.addWidget(self.corrosion_pixels)
        status2.addWidget(self.percent_label)
        
        status_layout.addLayout(status1)
        status_layout.addLayout(status2)        
        
        
        
        grid_layout.addLayout(selection_layout, 0, 0)
        grid_layout.addLayout(color_layout, 1, 0)
        grid_layout.addLayout(status_layout, 0, 1)
        grid_layout.addLayout(image_bar_layout, 1, 1)
        
        
        
        
        
        
        main_layout.addLayout(top_bar_layout)
        main_layout.addLayout(color_layout)
        main_layout.addLayout(image_bar_layout)
        main_layout.addLayout(bottom_bar_layout)
        widget = QWidget()
        #widget.setLayout(main_layout)
        widget.setLayout(grid_layout)
        self.setCentralWidget(widget)

    def choose_params(self):
        self.status_label.setText("检测状态: 参数选择")

    

    def choose_model_type(self):

        self.choose_params()
        

        if self.select_model_type.currentText() == "FastFCN":
            self.yml_name = "configs/fastfcn/fastfcn_resnet50_os8_ade20k_480x480_120k.yml"
            
        elif self.select_model_type.currentText() == "DeepLabV3 (ResNet-50)":
            self.yml_name = "configs/deeplabv3/deeplabv3_resnet50_os8_voc12aug_512x512_40k.yml"
        
        elif self.select_model_type.currentText() == "DeepLabV3P (ResNet-50)":
            self.yml_name = "configs/deeplabv3+/deeplabv3p_resnet50_os8_optic_disc_512x512_1k_teacher.yml"
            
        elif self.select_model_type.currentText() == "PP-LiteSeg":
            self.yml_name = "configs/pp-liteseg/pp_liteseg_optic_disc_512x512_1k.yml"
            
        elif self.select_model_type.currentText() == "FCN (HRNet-W18)":
            self.yml_name = "configs/fcn/fcn_hrnetw18_voc12aug_512x512_40k.yml"
            
        elif self.select_model_type.currentText() == "FCN (HRNet-W48)":
            self.yml_name = "configs/fcn/fcn_hrnetw48_voc12aug_512x512_40k.yml"
            
        elif self.select_model_type.currentText() == "SegFormer":
            self.yml_name = "configs/segformer/segformer_b1_cityscapes_1024x1024_160k.yml"
            
        elif self.select_model_type.currentText() == "PSPNet(ResNet-101)":
            self.yml_name = "configs/pspnet/pspnet_resnet101_os8_voc12aug_512x512_40k.yml"
        
        
        self.model_type_label.setText(f'模型类型: {self.select_model_type.currentText()}')
        
        self.model_type_label.setAlignment(Qt.AlignCenter)


    def choose_model(self):
        
        self.choose_params()
        self.model_filename = QFileDialog.getOpenFileName()[0]
        
        def cutstr(string):
            while(string.find("/") != -1):
                string = string[string.find("/") + 1 :]
            return string
        
        model_name = cutstr(self.model_filename)
        self.model_name_label.setText(f'模型文件: {model_name}')
        self.model_name_label.setAlignment(Qt.AlignCenter)
        
        
    def choose_source_image(self):
        
        self.choose_params()
        self.source_filename = QFileDialog.getOpenFileName()[0]
        print(self.source_filename)
        self.source_image_data = cv2.imread(self.source_filename)
        source_image_resized = resize_image(self.source_image_data, self.max_img_width, self.max_img_height)
        self.source_image.setPixmap(pixmap_from_cv_image(source_image_resized))
        
        
        self.image_size.setText(f'图像面积: {(self.source_image_data.shape[0]*self.source_image_data.shape[1])} 像素')
        

    def process_image(self):
        if self.select_model_type.currentText() == "点击选择模型类型":
            error_dialog = QErrorMessage()
            error_dialog.showMessage('请先选择模型类型')
            error_dialog.exec()
        
        elif self.model_filename is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('请先选择模型文件')
            error_dialog.exec()
        
        
        elif self.source_image_data is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('未检测到待检图像，请先导入图像')
            error_dialog.exec()
        else:
            
            self.setWindowTitle("船舶腐蚀检测与评估系统----检测中")
            
            
            def cutstr(string):
                while(string.find("/") != -1):
                    string = string[string.find("/") + 1 :]
                return string
            
            cfg = self.yml_name
            model_path = self.model_filename
            save_dir = "output/result"
            image_path = self.source_filename
            custom_color = []
            custom_color.append(self.back_r_select.value())
            custom_color.append(self.back_g_select.value())
            custom_color.append(self.back_b_select.value())
            custom_color.append(self.fore_r_select.value())
            custom_color.append(self.fore_g_select.value())
            custom_color.append(self.fore_b_select.value())
            
            
            
            c_pixel, percent = predict1(cfg, model_path, save_dir, image_path, custom_color)
            

            path = "output/result/added_prediction/"
            path += cutstr(self.source_filename)
            self.result_image_data =cv2.imread(path)
            self.percent_label.setText(f'腐蚀占比: {(percent * 100):.3f}%')
            self.percent_label.setAlignment(Qt.AlignCenter)
            self.corrosion_pixels.setText(f'腐蚀面积: {c_pixel} 像素')
            self.corrosion_pixels.setAlignment(Qt.AlignCenter)
            
            
            
            result_image_resized = resize_image(self.result_image_data, self.max_img_width, self.max_img_height)
            self.result_image.setPixmap(pixmap_from_cv_image(result_image_resized))
            
            self.setWindowTitle("船舶腐蚀检测与评估系统")
            self.status_label.setText("检测状态: 检测完成")

        
        
    def save_as_file(self):
        if self.result_image_data is None:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('未检测到分割输出，请先运行图像分割')
            error_dialog.exec()
        else:
            filename = QFileDialog.getSaveFileName(self, 'Save File')[0]
            if len(filename) > 0:
                cv2.imwrite(filename, self.result_image_data)
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("宋体")
    font.setPointSize(12)
    app.setFont(font)

    window = MainWindow()
    window.show()
    window.move(50,50)

    sys.exit(app.exec_())