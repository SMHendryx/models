
# coding: utf-8

# # DeepLab Demo
# 
# This demo will demostrate the steps to run deeplab semantic segmentation model on sample input images.

# In[4]:


#@title Imports

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

# In[5]:


#@title Helper methods


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image: PIL.Image) -> Tuple[PIL.Image.Image, numpy.ndarray]:
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.asarray([
      [128, 64, 128],
      [244, 35, 232],
      [70, 70, 70],
      [102, 102, 156],
      [190, 153, 153],
      [153, 153, 153],
      [250, 170, 30],
      [220, 220, 0],
      [107, 142, 35],
      [152, 251, 152],
      [70, 130, 180],
      [220, 20, 60],
      [255, 0, 0],
      [0, 0, 142],
      [0, 0, 70],
      [0, 60, 100],
      [0, 80, 100],
      [0, 0, 230],
      [119, 11, 32],
  ])
  return colormap



def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = get_dataset_colormap.label_to_color_image(seg_map, dataset = 'cityscapes').astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  # abstract this into function:
  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


PASCAL_LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

LABEL_NAMES = np.asarray([      'unlabeled'            , 
      'ego vehicle'          , 
      'rectification border' , 
      'out of roi'           , 
      'static'               , 
      'dynamic'              , 
      'ground'               , 
      'road'                 ,  
      'sidewalk'             , 
      'parking'              ,  
      'rail track'           ,
      'building'             , 
      'wall'                 , 
      'fence'                , 
      'guard rail'           , 
      'bridge'               , 
      'tunnel'               , 
      'pole'                 ,
      'polegroup'            , 
      'traffic light'        ,
      'traffic sign'         ,
      'vegetation'           ,   'terrain'              , 
      'sky'                  , 
      'person'               , 
      'rider'                , 
      'car'                  , 
      'truck'                , 
      'bus'                  , 
      'caravan'              , 
      'trailer'              , 
      'train'                , 
      'motorcycle'           ,   'bicycle'              ,  'license plate' ])

FULL_LABEL_MAP = np.arange(len(CITYSCAPES_LABEL_NAMES)).reshape(len(CITYSCAPES_LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP)


# In[6]:


#@title Select and download models {display-mode: "form"}

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

download = True
if download:
  download_path = os.path.join(model_dir, _TARBALL_NAME)
  print('downloading model, this might take a while...')
  urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
  print('download completed! loading DeepLab model...')

#already_downloaded = '/home/sean/repositories/tensorflow_models/models/research/deeplab/checkpoints/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz'
MODEL = DeepLabModel(download_path)#or: already_downloaded
print('model loaded successfully!')


# ## Run on sample images
# 
# Select one of sample images (leave `IMAGE_URL` empty) or feed any internet image
# url for inference.
# 
# Note that we are using single scale inference in the demo for fast computation,
# so the results may slightly differ from the visualizations in
# [README](https://github.com/tensorflow/models/blob/master/research/deeplab/README.md),
# which uses multi-scale and left-right flipped inputs.

# In[7]:


#@title Run on sample images {display-mode: "form"}

SAMPLE_IMAGE = 'image1'  # @param ['image1', 'image2', 'image3']
IMAGE_URL = ''  #@param {type:"string"}

_SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
               'deeplab/g3doc/img/%s.jpg?raw=true')


def run_visualization(url):
  """Inferences DeepLab model and visualizes result."""
  try:
    f = urllib.request.urlopen(url)
    jpeg_str = f.read()
    orignal_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return

  print('running deeplab on image %s...' % url)
  resized_im, seg_map = MODEL.run(orignal_im)

  vis_segmentation(resized_im, seg_map)


def runVisualization(path):
  """
  Inferences DeepLab model on local image and visualizes result
  Sean Hendryx
  """
  try:
    # load image
    orignal_im = Image.open(path)
  except IOError:
    print('Cannot retrieve image. Please check path: ' + path)
    return None

  print('running deeplab on image %s...' % path)
  resized_im, seg_map = MODEL.run(orignal_im)

  vis_segmentation(resized_im, seg_map)

image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE


image_path = '/Users/seanmhendryx/Explorer.ai/data/semantic_segmentation/images/two/frame0113.jpg'
runVisualization(image_path)



import numpy as np
import PIL.Image as img
import tensorflow as tf

from utils import get_dataset_colormap


def save_annotation(label,
                    save_dir,
                    filename,
                    add_colormap=True,
                    colormap_type=get_dataset_colormap.get_cityscapes_name()):
  """Saves the given label to image on disk.

  Args:
    label: The numpy array to be saved. The data will be converted
      to uint8 and saved as png image.
    save_dir: The directory to which the results will be saved.
    filename: The image filename.
    add_colormap: Add color map to the label or not.
    colormap_type: Colormap type for visualization.
  """
  # Add colormap for visualizing the prediction.
  if add_colormap:
    colored_label = get_dataset_colormap.label_to_color_image(
        label, colormap_type)
  else:
    colored_label = label

  pil_image = img.fromarray(colored_label.astype(dtype=np.uint8))
  with tf.gfile.Open('%s/%s.png' % (save_dir, filename), mode='w') as f:
    pil_image.save(f, 'PNG')


def getParentDir(filepath: str) -> str:
  """
  """
  # Get parent dir:
  import os.path
  return os.path.abspath(os.path.join(filepath, os.pardir))

def getBasenameNoExtension(filepath: str) -> str:
  """
  """
  from os.path import basename, splitext
  return splitext(basename(filepath))[0]

def savePredictions(model: DeepLabModel, input_path: str) -> None:
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

  # Get parent dir:
  par_dir = getParentDir(input_path)
  output_name = getBasenameNoExtension(input_path) + '_DeepLab_predictions'

  save_annotation(seg_map, par_dir, output_name)

savePredictions(MODEL, image_path)

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

  # make save dir if not set:
  if save_dir == None:
    save_dir = getParentDir(input_path)

  output_name = os.path.join(save_dir, getBasenameNoExtension(input_path) + '_DeepLab_predictions_overlay.png')

  plt.ioff()
  plt.figure()
  plt.imshow(resized_im)
  seg_im = get_dataset_colormap.label_to_color_image(seg_map, dataset = 'cityscapes').astype(np.uint8)
  plt.imshow(seg_im, alpha=0.7)
  plt.axis('off')
  plt.savefig(output_name, bbox = 'tight')


saveOverlay(MODEL, image_path) #, save_dir = '/Users/seanmhendryx/Explorer.ai/data/semantic_segmentation/images/two')

