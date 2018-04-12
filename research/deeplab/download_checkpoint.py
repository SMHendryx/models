# Downloads a TensorFlow model checkpoint to the download_checkpoint_to path


from utils.train_utils import download_xception_cityscapes_trainfine_checkpoint, download_checkpoint
#import tensorflow as tf
import argparse
import os


def parseArgs():
  # Returns an argparse.parse_args() object

  parser = argparse.ArgumentParser(description='Downloads the specified model checkpoint to the path --download_checkpoint_to.')
  parser.add_argument('--download_checkpoint_to', '-d', type=str, default = os.getcwd(), help='Path in which downloaded model checkpoint will be written.') # currently set up to only take one file.
  parser.add_argument('--model', '-m', type=str, default='xception_cityscapes_trainfine', help="Model to download. Can be one of: @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval'].")
  args = parser.parse_args()
  return args

def main():

  args = parseArgs()
  print('Downloading model ' + args.model)
  download_checkpoint(model_name = args.model, model_dir = args.download_checkpoint_to)



if __name__ == '__main__':
  main()
