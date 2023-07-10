import os
import shutil

##############
# Constants
##############
SOURCE_DIR = '.data'
'''The dataset should be downloaded in SOURCE_DIR'''

DEST_DIR = SOURCE_DIR
'''Output directory for train/test split'''

DEST_TRAIN = f'{DEST_DIR}/train/images'
DEST_TEST = f'{DEST_DIR}/test/images'

LABELS = ['glide', 'loose', 'none', 'slab']


if __name__ == '__main__':
    '''Generate train/test split for the image classification task'''

    # Check that train/test split txt files exist
    train_split = os.path.join(SOURCE_DIR, 'train.txt')
    test_split = os.path.join(SOURCE_DIR, 'test.txt')
    assert os.path.exists(train_split), f'File not found: {train_split}'
    assert os.path.exists(test_split), f'File not found: {test_split}'

    # Read image paths for train/test split
    with open(train_split) as f:
        train_ims = [line.rstrip('\n') for line in f]
    with open(test_split) as f:
        test_ims = [line.rstrip('\n') for line in f]

    # Delete and re-create output directories
    for train_test in [DEST_TRAIN, DEST_TEST]:
        try:
            shutil.rmtree(train_test)
        except:
            pass
        for l in LABELS:
            os.makedirs(os.path.join(train_test, l), exist_ok=False)

    # Copy images into destination directory
    print('Generating train/test split...')
    ims_directory = os.path.join(SOURCE_DIR, 'images')
    for root, dirnames, filenames in os.walk(ims_directory):
        for file_name in filenames:
            assert file_name.lower().endswith(
                '.jpg'), f'Unexpected file name {file_name}'

            # Get whether file is in the train or test split
            path_tail = '/' + root.split('/')[-1] + f'/{file_name}'
            target_dir = None
            if path_tail in train_ims:
                target_dir = DEST_TRAIN
            elif path_tail in test_ims:
                target_dir = DEST_TEST
            else:
                raise FileNotFoundError(
                    f'Image {path_tail} not in train or test split')

            # Copy image to destination directory
            dest_file = target_dir + path_tail
            shutil.copyfile(f'{root}/{file_name}', dest_file)

    print('Done')
