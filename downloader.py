# python3
# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Open Images image downloader.

This script downloads a subset of Open Images images, given a list of image ids.
Typical uses of this tool might be downloading images:
- That contain a certain category.
- That have been annotated with certain types of annotations (e.g. Localized
Narratives, Exhaustively annotated people, etc.)

The input file IMAGE_LIST should be a text file containing one image per line
with the format <SPLIT>/<IMAGE_ID>, where <SPLIT> is either "train", "test",
"validation", or "challenge2018"; and <IMAGE_ID> is the image ID that uniquely
identifies the image in Open Images. A sample file could be:
  train/f9e0434389a1d4dd
  train/1a007563ebc18664
  test/ea8bfd4e765304db

"""

import argparse
from concurrent import futures
import os
import re
import sys

import boto3
import botocore
import tqdm

BUCKET_NAME = 'open-images-dataset'
REGEX = r'^(test|train|validation|challenge2018)/([a-fA-F0-9]+)$'


def check_and_homogenize_one_image(image):
  match = re.fullmatch(REGEX, image)
  if not match:
    raise ValueError(f'Image string is not recognized: "{image}".')
  split, image_id = match.groups()
  return split, image_id


def check_and_homogenize_image_list(image_list):
  for line_number, image in enumerate(image_list):
    try:
      yield check_and_homogenize_one_image(image)
    except (ValueError, AttributeError):
      raise ValueError(
          f'ERROR in line {line_number + 1} of the image list. The following image '
          f'string is not recognized: "{image}".')


def read_image_list_file(image_list_file):
  with open(image_list_file, 'r') as f:
    for line in f:
      value = line.strip()
      if not value or value.startswith('#'):
        continue
      if value.lower().endswith('.jpg'):
        value = value[:-4]
      yield value


def download_one_image(bucket, split, image_id, download_folder):
  output_path = os.path.join(download_folder, f'{image_id}.jpg')
  if os.path.exists(output_path):
    return True, f'{split}/{image_id} (already exists)'

  try:
    bucket.download_file(f'{split}/{image_id}.jpg', output_path)
    return True, None
  except botocore.exceptions.ClientError as exception:
    return False, (
        f'ERROR when downloading image `{split}/{image_id}`: {str(exception)}')


def download_all_images(args):
  """Downloads all images specified in the input file."""
  bucket = boto3.resource(
      's3', config=botocore.config.Config(
          signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)

  download_folder = args['download_folder'] or os.getcwd()

  if not os.path.exists(download_folder):
    os.makedirs(download_folder)

  try:
    total_images = sum(
        1 for _ in check_and_homogenize_image_list(
            read_image_list_file(args['image_list'])))
  except ValueError as exception:
    sys.exit(exception)

  failures = []
  max_workers = max(1, args['num_processes'])
  max_pending = max_workers * 4

  progress_bar = tqdm.tqdm(total=total_images, desc='Downloading images', leave=True)
  with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    pending_futures = set()

    try:
      image_iter = check_and_homogenize_image_list(read_image_list_file(args['image_list']))
      for split, image_id in image_iter:
        pending_futures.add(
            executor.submit(download_one_image, bucket, split, image_id, download_folder))

        if len(pending_futures) >= max_pending:
          done, pending_futures = futures.wait(
              pending_futures, return_when=futures.FIRST_COMPLETED)
          for future in done:
            success, error_message = future.result()
            if not success:
              failures.append(error_message)
            progress_bar.update(1)
    except ValueError as exception:
      sys.exit(exception)

    for future in futures.as_completed(pending_futures):
      success, error_message = future.result()
      if not success:
        failures.append(error_message)
      progress_bar.update(1)
  progress_bar.close()

  if failures:
    print(f'Finished with {len(failures)} download failures.', file=sys.stderr)
    for error_message in failures[:20]:
      print(error_message, file=sys.stderr)
    if len(failures) > 20:
      print(f'... plus {len(failures) - 20} more failures.', file=sys.stderr)
    sys.exit(1)

  print(f'Successfully downloaded {total_images} images.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      'image_list',
      type=str,
      default=None,
      help=('Filename that contains the split + image IDs of the images to '
            'download. Check the document'))
  parser.add_argument(
      '--num_processes',
      type=int,
      default=5,
      help='Number of parallel processes to use (default is 5).')
  parser.add_argument(
      '--download_folder',
      type=str,
      default=None,
      help='Folder where to download the images.')
  download_all_images(vars(parser.parse_args()))
