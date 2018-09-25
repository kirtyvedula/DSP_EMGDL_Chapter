import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, model_from_json
from keras import activations
from vis.visualization import visualize_cam, visualize_saliency, get_num_filters
from vis.utils import utils
from vis.input_modifiers import Jitter
import scipy.io
import os
import sys
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from utils import *
from generator import *
import json
import re
import math

def stitch_images(images, margin=5, cols=5, pad_fill='black'):
    """Utility function to stitch images together with a 'margin'.
    Args:
        images: The array of 2D images to stitch
        margin: The margin size between images (Default value = 5)
        cols: Max number of image cols. New row is created when number of images exceed the column size.
            (Default value = 5)
        pad_fill: The color of margin ('black' oe 'white')
    Returns:
        A single numpy image array comprising of input images.
    """
    if len(images) == 0:
        return None
    h, w, c = images[0].shape
    n_rows = int(math.ceil(len(images) / cols))
    n_cols = min(len(images), cols)
    out_w = n_cols * w + (n_cols - 1) * margin
    out_h = n_rows * h + (n_rows - 1) * margin
    images = images.astype('float64')
    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    if pad_fill == 'black':
        stitched_images = np.zeros((out_h, out_w, c), dtype='float64')
    else:
        stitched_images = np.ones((out_h, out_w, c), dtype='float64')
    for row in range(n_rows):
        for col in range(n_cols):
            img_idx = row * cols + col
            # img_h, img_w = images[img_idx].shape[:2]
            if img_idx >= len(images):
                break
            stitched_images[(h + margin) * row: (h + margin) * row + h,
                            (w + margin) * col: (w + margin) * col + w, :] = images[img_idx]
    return stitched_images


def save_stitched(imgs, cols, pad_fill='black', path='', border=True):
    """Utility function to save stitched images.
    Args:
        imgs: The np.ndarray of stitched images
        cols: Max number of image columns used for stitching
        pad_fill: The color of margin ('black' oe 'white')
        path: The path where the image is saved
        border: Whether a black border is addded around the image
    """
    fig = plt.figure()
    stitched = stitch_images(imgs, 2, cols, pad_fill)
    if stitched.shape[-1] == 1:
        stitched = np.stack((np.squeeze(stitched),) * 3, axis=-1)
    if border is False:
        plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(stitched, interpolation='none')
    plt.savefig(path)


def get_model(models_path='.', model_name='', subject=20):
    """Utility function that finds the model files.
    Args:
        models_path: Folder path where models are kept
        model_name: Name of the model to retrieve. The format of the files are: '*(model_name)_(subject)
        subject: The subject of whom the model is returned
    Returns:
        A keras.model
    """
    filename = ''
    for file in os.listdir(models_path):
        match = re.match(r'(.*{}_{})\.(?:json|h5)'.format(model_name, subject), file)
        if match:
            filename = match.group(1)
            json_filename = os.path.join(models_path, filename + '.json')
            json_file = open(json_filename, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            weights_filename = os.path.join(models_path, filename + '.h5')
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(weights_filename)
            break

    print('Found model {}'.format(filename))
    return model



def saliency(model, input_images, input_labels):
    """Function that computes the attention map visualization.
    Args:
        model: A keras.model
        input_images: Array of 3D images (height, width, 3) of which the attention is computed
        input_labels: The class label for each input image
    Returns:
        A list of attention maps
    """
    layer_idx = -1

    # Swap softmax with linear
    model.layers[-2].activation = activations.linear
    model = utils.apply_modifications(model)

    # This is the output node we want to maximize.
    vis_images = []
    for l in range(len(input_images)):
        img = input_images[l]
        label = input_labels[l]
        grads = visualize_saliency(model, layer_idx, filter_indices=label,
                                   seed_input=img)
        vis_images.append(grads)
    return vis_images


def vis_filters(model, layer_names):
    """Function that retrieves the filters of model layers.
    Args:
        model: A keras.model
        layer_names: List of layer names of which the filters are returned
    Returns:
        A list of layer filters
    """
    filters = []
    for layer_name in layer_names:
        layer_idx = utils.find_layer_idx(model, layer_name)
        conv_layer = model.layers[layer_idx]
        ws = conv_layer.get_weights()[0]
        ws = np.rollaxis(ws, 3)
        ws = np.reshape(ws[:,:,:,0], ws.shape[:3])
        cmap = plt.get_cmap('bone')
        ws = cmap(ws)
        filters.append(ws)
    return filters


def occlusion(model, input_images, input_labels, val=0):
    """Function that computes the occlusion map.
    Args:
        model: A keras.model
        input_images: Array of 3D images (height, width, 3)
        input_labels: The class label for each input image
        val: Value used for occlusion (0 or 1)
    Returns:
        The occlusion map
    """
    enc = LabelBinarizer()
    enc.fit(input_labels.reshape((-1, 1)))
    input_images = np.array(input_images)

    occlusion_acc = []
    x = np.copy(input_images)
    predictions = model.predict(x)
    predictions = enc.inverse_transform(predictions.squeeze())
    cm = confusion_matrix(input_labels, predictions)
    occlusion_acc.append(np.diagonal(cm))

    for j in range(0, 10):
        x = np.copy(input_images)
        x[:, :, j, :] = val
        predictions = model.predict(x)
        predictions = enc.inverse_transform(predictions.squeeze())
        cm = confusion_matrix(input_labels, predictions)
        d = np.diagonal(cm)
        occlusion_acc.append(d)

    occlusion_acc = np.array(occlusion_acc)
    occlusion_acc = (occlusion_acc - np.min(occlusion_acc)) / (np.max(occlusion_acc) - np.min(occlusion_acc))
    cmap = plt.get_cmap('GnBu_r')
    occlusion_acc = cmap(occlusion_acc)
    occlusion_acc = np.delete(occlusion_acc, 3, 2)
    return occlusion_acc


np.random.seed(1992)

MODE = sys.argv[1]
MODELS_PATH = sys.argv[2]
OUT_PATH = sys.argv[3]
CONFIG_FILE = str(sys.argv[4])
SUBJECT = 11

with open(CONFIG_FILE) as json_file:
    config_data = json.load(json_file)

PARAMS_DATASET = config_data['dataset']
PARAMS_MODEL = config_data['model']
PARAMS_VALID_GENERATOR = DEFAULT_GENERATOR_PARAMS.copy()
params_gen = PARAMS_DATASET.get('valid_generator', {}).copy()
for key in params_gen.keys():
    PARAMS_VALID_GENERATOR[key] = params_gen[key]
PARAMS_VALID_GENERATOR['input_directory'] = 'dataset/Ninapro-DB1-Proc/subject-{:02d}'.format(SUBJECT)
PARAMS_VALID_GENERATOR['classes'] = [i for i in range(53)]
PARAMS_VALID_GENERATOR['repetitions'] = PARAMS_VALID_GENERATOR['repetitions'][0:1]
PARAMS_VALID_GENERATOR['shuffle'] = False

print(PARAMS_VALID_GENERATOR)
print('MODE: ', MODE)
print('MODELS_PATH: ', MODELS_PATH)
print('OUT_PATH: ', OUT_PATH)

if MODE == 'saliency':
    valid_generator = DataGenerator(**PARAMS_VALID_GENERATOR)
    X_test, Y_test, test_reps = valid_generator.get_data()
    Y_test = np.argmax(Y_test, axis=1)

    model = get_model(MODELS_PATH, PARAMS_MODEL['save_file'], SUBJECT)
    u = np.unique(Y_test)
    attention = []
    for i in range(53):
        imgs = X_test[np.isin(Y_test, u[i])]
        imgs = imgs[len(imgs) // 2 : len(imgs) // 2 + 1]
        print('Label: {}, Segments: {}'.format(u[i], len(imgs)))
        X = []
        Y = []
        for img in imgs:
            x = np.squeeze(img)
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))
            X.append(x)
            Y.append(int(u[i]))
        grads = saliency(model, X, Y)
        grads = np.concatenate(grads)
        attention.append(grads)
    save_stitched(np.array(attention), 10, 'white', '{}/{}_attention.jpg'.format(OUT_PATH, PARAMS_MODEL['save_file']))
    plt.imshow(attention[0])
    plt.show()
    plt.imshow(attention[1])
    plt.show()

elif MODE == 'filters':
    model = get_model(MODELS_PATH, PARAMS_MODEL['save_file'], SUBJECT)
    layer_names = ['b1_conv2d_32_1x10']
    filters = vis_filters(model, layer_names)
    print(len(filters))
    print(filters[0].shape)
    for i in range(len(filters)):
        save_stitched(filters[i], 1, 'white', '{}/{}_{}.jpg'.format(OUT_PATH, PARAMS_MODEL['save_file'], layer_names[i]))

elif MODE == 'occlusion':
    valid_generator = DataGenerator(**PARAMS_VALID_GENERATOR)
    X_test, Y_test, test_reps = valid_generator.get_data()
    Y_test = np.argmax(Y_test, axis=1)

    model = get_model(MODELS_PATH, PARAMS_MODEL['save_file'], SUBJECT)
    occ = occlusion(model, X_test, Y_test, val=0)
    plt.imshow(occ)
    plt.savefig('{}/{}_occlusion_0.jpg'.format(OUT_PATH, PARAMS_MODEL['save_file']))
    scipy.io.savemat('{}/{}_occlusion_0.mat'.format(OUT_PATH, PARAMS_MODEL['save_file']), {'occlusion_acc': occ})

    occ = occlusion(model, X_test, Y_test, val=1)
    plt.imshow(occ)
    plt.savefig('{}/{}_occlusion_1.jpg'.format(OUT_PATH, PARAMS_MODEL['save_file']))
    scipy.io.savemat('{}/{}_occlusion_1.mat'.format(OUT_PATH, PARAMS_MODEL['save_file']), {'occlusion_acc': occ})


