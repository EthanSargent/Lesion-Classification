import ipdb
import imageio
import cv2
import sys
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from numpy.random import shuffle
from collections import Counter

def get_multiclass_paths(descriptions_dir, raw_images_dir):
    image_IDs = []
    diagnoses = []
    for ID in listdir(descriptions_dir):
        with open (join(descriptions_dir, ID), "r") as file:
            data = file.read().replace('\n', '')
            try:
                diagnoses.append(json.loads(data)['meta']['clinical']['diagnosis'])
            except:
                continue
            
    cnt = Counter(diagnoses)
    del cnt[None]
    class_dict = {diagnosis:count for diagnosis, count in cnt.most_common(9)}
    classes = list(class_dict.keys())
    
    for ID in listdir(descriptions_dir):
        with open (join(descriptions_dir, ID), "r") as file:
            data = file.read().replace('\n', '')
            try:
                diagnosis = json.loads(data)['meta']['clinical']['diagnosis']
                if diagnosis in classes:
                    image_IDs.append(ID)
            except:
                continue
    image_IDs.sort()
    
    image_paths = [join(raw_images_dir, f) for f in listdir(raw_images_dir)\
                       if (f[:12] in image_IDs)]
    image_paths.sort()

    description_paths = [join(descriptions_dir, f) for f in image_IDs]
    
    # permuting the dataset removes some of the class imbalance among the chunks
    np.random.seed(20)
    X = np.asarray([description_paths, image_paths]).T
    shuffle(X)
    description_paths = X[:,0]
    image_paths= X[:,1]
    
    return description_paths, image_paths, class_dict

def get_paths(descriptions_dir, images_dir): 
    # we want a list of the IDs of images that have a useful binary label
    # as benign or malignant
    image_IDs = []
    for ID in listdir(descriptions_dir):
        with open (join(descriptions_dir, ID), "r") as file:
            data = file.read().replace('\n', '')
            try:
                json.loads(data)['meta']['clinical']['benign_malignant']
                image_IDs.append(ID)
            except:
                continue
    image_IDs.sort()
    
    # compute the list of image paths for all images that have useful
    # binary label (i.e., those in image_IDs [without .jpeg/.png extension])
    image_paths = [join(images_dir, f) for f in listdir(images_dir) if (f[:12] in image_IDs)]
    image_paths.sort()

    description_paths = [join(descriptions_dir, f) for f in image_IDs]
    
    return description_paths, image_paths

def process_chunks(chunks, data_dir, descriptions_dir, raw_images_dir, estimate_aspect_ratio):
    ''' Processes the raw data into 'chunks' number of image and label
        PyTorch tensors, to be stored in the 'data_dir' directory. If
        'estimate_aspect_ratio' is passed, a new estimate of the mean
        aspect ratio is computed prior to chunk loading. All images are
        reshaped to (216, 216/aspect_ratio) in order to standardize
        input to the neural network.
    '''
    if estimate_aspect_ratio:
        aspect_ratio = compute_estimated_aspect_ratio(image_filenames)
    else:
        aspect_ratio = 0.7105451408210631

    description_paths, image_paths = get_paths(descriptions_dir, raw_images)
    
    # permuting the dataset removes some of the class imbalance among the chunks
    np.random.seed(20)
    X = np.asarray([description_paths, image_file_paths]).T
    shuffle(X)
    description_paths = X[:,0]
    image_paths= X[:,1]
        
    n = len(description_paths)
    chunk_size = n//chunks
    
    for chunk in range(chunks):
        load_chunk(chunk, chunk_size, description_paths, image_paths, aspect_ratio)

# compute an estimate of the mean aspect ratio
def compute_estimated_aspect_ratio(image_filenames):
    ratios = []
    np.random.seed(20)
    sample = np.random.choice(image_filenames, 1000)
    for filename in sample:
        x = imageio.imread(filename)
        ratios.append(x.shape[0]/x.shape[1])
    aspect_ratio = np.mean(np.asarray(ratios))
    return aspect_ratio

def load_chunk(chunk, chunk_size, description_paths, image_paths, aspect_ratio):
    image_numbers = list(range(chunk*chunk_size, (chunk+1)*chunk_size))
    if chunk==9:
        image_numbers = list(range(chunk*chunk_size, n))

    X = torch.empty(size=(len(image_numbers), 216, int(216/aspect_ratio), 3))
    Y = []
    for sample_idx, idx in enumerate(tqdm(image_numbers)):
        # resize the images to the computed mean aspect ratio using cv2
        img = cv2.imread(image_paths[idx])
        res = cv2.resize(img, dsize=(int(216/aspect_ratio), 216), interpolation=cv2.INTER_CUBIC)
        X[sample_idx] = torch.tensor(res)
        with open (description_paths[idx], "r") as file:
            data = file.read().replace('\n', '')
            Y.append(json.loads(data)['meta']['clinical']['benign_malignant'])
    Y = torch.tensor([1 if diagnosis=='malignant' else 0 for diagnosis in Y])
    X = X.permute(0,3,1,2)
    print("Finished chunk " + str(chunk))
    torch.save(X, 'data/images-' + str(chunk) + '.pt')
    torch.save(Y, 'data/labels-' + str(chunk) + '.pt')

def confirm_arguments(args):
    print('You have decided to do the following:')
    if args.chunks is None:
        print('Process data in 10 chunks')
    else:
        print('Process data in {0} chunks'.format(args.chunks))

    if args.data_dir is None:
        print('Images and label tensors will be downloaded to "data" directory.')
    else:
        print('Images and label tensors will be downloaded to "{0}" directory.'.format(args.data_dir))

    res = input('Do you confirm your choices? [Y/n] ')

    while res not in ['y', '', 'n']:
        res = input('Invalid input. Do you confirm your choices? [Y/n] ')
    if res in ['y', '']:
        return True
    if res == 'n':
        return False

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', type=int,
                        help='The number of chunks into which the raw image dataset will be broken up.'
                        'The last chunk will be used for testing. If this argument is passed, the'
                        'the entire dataset will be re-partitioned.', default=10)
    parser.add_argument('--data_dir', type=str,
                        help='The directory into which the image and label tensors will be downloaded.',\
                        default='data')
    parser.add_argument('--raw_images_dir', type=str,
                        help='Where the raw images are located.',\
                        default=join('raw_data','Images'))
    parser.add_argument('--descriptions_dir', type=str,
                        help='Where the verbose image descriptions are located.',\
                        default=join('raw_data','Descriptions'))
    parser.add_argument('--aspect_ratio', help='Whether to recompute the mean aspect ratio.', action="store_true")
    parsed_args = parser.parse_args(args)
    return parsed_args

def main(args):
    args = parse_args(args)
    has_confirmed = confirm_arguments(args)

    if has_confirmed:
        process_chunks(args.chunks, args.data_dir, args.descriptions_dir, args.raw_images_dir,\
                       estimate_aspect_ratio = args.aspect_ratio)
    else:
        print('Exiting without downloading anything')

if __name__ == "__main__":
    main(sys.argv[1:])



