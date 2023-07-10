import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

from np_image import NpImage

LABELS = ['glide', 'loose', 'none', 'slab']


def count_images(data_dir: str) -> int:
    '''Recursively crawl the data directory and print statistics on
        - the total number of images
        - the number of images for each label
        - the number of distinct locations

    Args:
        data_dir (str): the directory to crawl.

    Returns:
        (int) the total number of images.'''

    assert os.path.isdir(
        data_dir), f'Path not found: {os.path.abspath(data_dir)}'

    labels = ['loose', 'none', 'glide', 'slab']
    label_extensions = {}
    total_jpgs: int = 0
    im_locations:  dict[str, int] = {}

    # Count number of .jpg files in each label subdirectory
    for l in labels:
        extensions: dict[str, int] = {}
        label_dir = os.path.join(data_dir, l)
        for root, _, file_names in os.walk(label_dir):
            for f_name in file_names:
                # Check that file name matches the regex format
                #                Y    Y    Y    Y  -  M    M  -  D    D    [     location              ]         (v)
                file_regex = r'[1-2][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]\s[a-z\s0-9\-\.,]+[a-z0-9\-\.,](\s\([0-9]+\))?.jpg\Z'
                assert bool(re.match(file_regex, f_name)
                            ), f'Unexpected file name format: {f_name}'

                # Save file extension metadata
                file_name, file_ext = os.path.splitext(f_name)
                extensions[file_ext] = 1 + extensions.get(file_ext, 0)

                # Remove date from file name
                fname_without_date = re.sub(
                    r'[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]\s', '', file_name)
                # Extract location from file name
                location = re.match(
                    r'[a-zA-Z\s0-9\-\.,]+\s\(', fname_without_date).group(0)[:-2]
                im_locations[location] = 1 + im_locations.get(location, 0)

        label_extensions[l] = extensions.get('.jpg', 0)
        total_jpgs += extensions.get('.jpg', 0)

    print(f'Total Images: {total_jpgs}')
    print('Label Distribution (of .jpg images):')
    print(f'    {label_extensions}')
    print(
        f'Number of distinct locations: {len(im_locations.keys())} locations over {sum(im_locations.values())} images')

    return total_jpgs


def check_train_test_split(train_dir: str, test_dir: str, n_ims: int):
    '''Analyse image dimensions and print out the mean and standard deviation in px

    Args:
        n_ims (int): the total number of images.

    Returns:
        None.
    '''
    def _walk_dir(data_dir: str) -> Tuple[List[str], List[str], List[str]]:
        '''Walks a directory and returns a tuple of the image paths, labels, and locations'''
        im_paths: List[str] = []
        im_labels: List[str] = []
        im_locations: List[str] = []

        for root, _, filenames in os.walk(data_dir):
            for file_name in filenames:
                if not (file_name.lower().endswith('.jpg')):
                    continue

                # Find label in path
                label = None
                for l in LABELS:
                    if f'/{l}' in root:
                        label = l
                assert label is not None, f'Could not deduce label from path {root}'

                file_name_without_date = file_name[11:]
                location = re.search(
                    r'[a-z\s0-9\-\.,]+[a-z0-9\-\.,]', file_name_without_date).group(0)
                file_path = os.path.join(root, file_name)

                im_paths.append(file_path)
                im_labels.append(label)
                im_locations.append(location)
        return im_paths, im_labels, im_locations

    # Run some sanity checks
    test_paths, test_labels, test_locations = _walk_dir(test_dir)
    train_paths, train_labels, train_locations = _walk_dir(train_dir)
    assert (len(train_paths) + len(test_paths)
            ) == n_ims, f'Expected {n_ims} images but got {len(train_paths)} + {len(test_paths)}'

    # Check that train and test locations do not overlap
    location_overlap = list(set(test_locations) & set(train_locations))
    assert location_overlap == [
    ], f'Train and test set locations overlap: {location_overlap}'

    print(
        f'Found {len(train_paths)} train images and {len(test_paths)} test images')


def check_image_annotations(data_dir: str, ims_dir: str):
    '''Analyse the distribution of image annotations and run some simple sanity checks.

    Args:
        data_dir (str): the directory to crawl.
        ims_dir (str): the directory containing image files.

    Returns:
        None.
    '''
    labels = ['loose', 'glide', 'slab']
    file_counts: Dict[str, int] = {}
    region_counts: Dict[str, int] = {}
    num_multilabel_ims = 0

    # Count number of .json files in each annotation label subdirectory
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for root, _, file_names in os.walk(label_dir):
            for f_name in file_names:
                # Check file extension
                _, file_ext = os.path.splitext(f_name)
                assert file_ext == '.json', f'Unexpected file format {f_name}'

                # Add all .json files in the directory to file_counts
                file_counts[label] = 1 + file_counts.get(label, 0)

                # Add annotated regions to region_counts
                with open(os.path.join(root, f_name), 'rb') as ann_f:
                    ann_obj = json.load(ann_f)

                ann_w = ann_obj['asset']['size']['width']
                ann_h = ann_obj['asset']['size']['height']
                im_tags: Dict[str, int] = {}

                im_region_labels = sum([r['tags']
                                       for r in ann_obj['regions']], [])
                if len(list(set(im_region_labels))) > 1:
                    num_multilabel_ims += 1

                for r in ann_obj['regions']:
                    region_tags: List[str] = r['tags']
                    assert len(
                        region_tags) == 1, f'Expected 1 tag per region but got {region_tags}'
                    tag = region_tags[0]
                    im_tags[tag] = 1 + im_tags.get(tag, 0)
                    region_counts[tag] = 1 + region_counts.get(tag, 0)

                    # Check that bounding box is contained within image
                    bb_left = r['boundingBox']['left']
                    bb_top = r['boundingBox']['top']
                    bb_right = int(bb_left + r['boundingBox']['width'])
                    bb_bottom = int(bb_top + r['boundingBox']['height'])

                    bbox_err_str = f'for {ann_w}x{ann_h} image {root}/{f_name}'
                    assert (bb_left >= 0 and bb_left <=
                            ann_w), f'Bad bounding box left coordinate {bb_left} {bbox_err_str}'
                    assert (bb_top >= 0 and bb_top <=
                            ann_h), f'Bad bounding box top coordinate {bb_top} {bbox_err_str}'
                    assert (bb_right >= 0 and bb_right <=
                            ann_w), f'Bad bounding box right coordinate {bb_right} {bbox_err_str}'
                    assert (bb_bottom >= 0 and bb_bottom <=
                            ann_h), f'Bad bounding box bottom coordinate {bb_bottom} {bbox_err_str}'

                    # Check that annotation bounding box coordinates are all within the image
                    for p in r['points']:
                        x = int(p['x'])
                        y = int(p['y'])
                        assert (x >= 0 and x <= ann_w and y >= 0 and y <=
                                ann_h), f'Can coordinate {p} {bbox_err_str}'

                # If a single image contains multiple types of avalanche then analyse further
                if len(im_tags) > 1:
                    # Check that the image label is in the image tags (sanity check)
                    assert label in im_tags.keys(
                    ), f'Label {label} not in {im_tags}. Labelling error?'

                assert ann_obj['asset']['format'] == 'jpg', 'Unexpected annotation file format value'
                assert ann_obj['asset']['name'] in ann_obj['asset']['path'], 'Bad annotation file name'

                # Assert that annotation image dimensions match actual dimensions
                im_path = ims_dir + ann_obj['asset']['path']
                true_w, true_h = NpImage(
                    im_path=im_path).get_PIL_image_exif().size
                assert (ann_w == true_w and ann_h ==
                        true_h), f'Bad image dimensions {(ann_w,ann_h)}, {(true_w,true_h)}'

    # Print number of annotations
    print(f'{sum(file_counts.values())} annotation files found with labels {file_counts}.')
    print(f'{sum(region_counts.values())} annotated regions found with tags {region_counts}.')
    print(f'{num_multilabel_ims} multilabel images')


if __name__ == '__main__':
    DATA_DIR = '.data'
    IMS_DIR = os.path.join(DATA_DIR, 'images')
    ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')
    TRAIN_SPLIT = os.path.join(DATA_DIR, 'train.txt')
    TEST_SPLIT = os.path.join(DATA_DIR, 'test.txt')

    # Assert that all expected files exist
    for dir in [DATA_DIR, IMS_DIR, ANNOTATION_DIR]:
        assert os.path.isdir(
            dir), f'Directory not found: {os.path.abspath(dir)}'
    for path in [TRAIN_SPLIT, TEST_SPLIT]:
        assert os.path.exists(path), f'Path not found: {os.path.abspath(dir)}'

    # Output number of files for each label
    n_ims = count_images(IMS_DIR)
    print('-'*50)

    # If train/test folders exist, then check these
    train_dir = f'{DATA_DIR}/train'
    test_dir = f'{DATA_DIR}/test'
    if os.path.exists(train_dir) or os.path.exists(test_dir):
        print('Checking train/test split...')
        check_train_test_split(f'{DATA_DIR}/train', f'{DATA_DIR}/test', n_ims)
        print('-'*50)

    # Analyse image annotations
    print('Checking image annotations... (may take a few minutes)')
    check_image_annotations(ANNOTATION_DIR, IMS_DIR)
