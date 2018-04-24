# Runs DeepLab inference on all images in directory
# Sean Hendryx forked from: https://github.com/tensorflow/models/blob/master/research/deeplab/
# coding: utf-8


import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

from utils import get_dataset_colormap
from typing import Tuple, List, Dict

import logging
import sys
import DeepLabModel

import argparse

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
#Set up file handler:
#fh = logging.FileHandler(os.path.basename(__file__) + str(datetime.now()).replace(" ", "_") + '.log')
# Pipe output to stdout
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


def parseArgs():
  # Returns an argparse.parse_args() object

  parser = argparse.ArgumentParser(description='Runs DeepLab inference on all images in directory')
  parser.add_argument('--inputDir', '-i', type=str, required = True, help='Path to directory of images to run inference on.')
  parser.add_argument('--ouputDir', '-o', type=str, required = False, default = None, help='Path to write output to. Default will write in same directory as input.') # currently set up to only take one file.

  args = parser.parse_args()

  return args


def main():

  args = parseArgs()

  input_dir = args.inputDir
  ouput_dir = args.ouputDir

  checkExists(input_dir)

  logger.info("Running DeepLab inference on: %s" % input_dir)
  model = DeepLabModel(downloadDeepLabModel())
  deeplabVizInferAllImages(model, input_dir, output_dir)


def checkExists(file: str) -> None:
  """
  Check if file or directory exists
  """
  if not os.path.exists(file):
    raise FileNotFoundError(logger.error("Input path %s does not exist." % file))


def downloadDeepLabModel() -> str:
  """
  Returns path to downloaded tarball to temp location
  """
  MODEL_NAME = 'xception_cityscapes_trainfine'  # @param ['xception_cityscapes_trainfine', 'mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']
  _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
  _MODEL_URLS = {
        'xception_cityscapes_trainfine':
            'deeplabv3_cityscapes_train_2018_02_06.tar.gz',
      'mobilenetv2_coco_voctrainaug':
          'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
      'mobilenetv2_coco_voctrainval':
          'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
      'xception_coco_voctrainaug':
          'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
      'xception_coco_voctrainval':
          'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
  }
  _TARBALL_NAME = 'deeplab_model.tar.gz'

  model_dir = tempfile.mkdtemp()
  tf.gfile.MakeDirs(model_dir)

  download_path = os.path.join(model_dir, _TARBALL_NAME)
  logger.info('downloading model, this might take a while...')
  urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
  logger.info('download completed!')


def deeplabVizInferAllImages(model: DeepLabModel, input_dir:str, output_dir:str = None) -> None:
  """
  """
  # make save dir if not set:
  if output_dir == None:
    output_dir = getParentDir(input_path)

  for input_image in os.listdir(input_dir):
    deeplabVizInfer(input_image, output_dir)

def getParentDir(filepath: str) -> str:
  """
  """
  # Get parent dir:
  import os.path
  return os.path.abspath(os.path.join(filepath, os.pardir))


def deeplabVizInfer(model: DeepLabModel, input_image_path:str, output_dir:str) -> None:
  """
  """
  saveOverlay(model, input_image_path, output_dir)


def saveOverlay(model: DeepLabModel, input_path: str, save_dir: str = None) -> None:
  """
  Inferences DeepLab model on local image and visualizes result
  Sean Hendryx
  :param model: DeepLab model
  :param input_path: path to input image
  :return: None
  """
  try:
    # load image
    orignal_im = Image.open(input_path)
  except IOError:
    print('Cannot retrieve image. Please check path: ' + input_path)
  return None

  print('running deeplab on image %s...' % input_path)
  resized_im, seg_map = model.run(orignal_im)

  output_name = os.path.join(save_dir, getBasenameNoExtension(input_path) + '_DeepLab_predictions_overlay.png')

  plt.ioff()
  plt.figure()
  plt.imshow(resized_im)
  seg_im = get_dataset_colormap.label_to_color_image(seg_map, dataset = 'cityscapes').astype(np.uint8)
  plt.imshow(seg_im, alpha=0.7)
  plt.axis('off')
  plt.savefig(output_name, bbox = 'tight')

def getBasenameNoExtension(filepath: str) -> str:
  """
  """
  from os.path import basename, splitext
  return splitext(basename(filepath))[0]

def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':
    main()

