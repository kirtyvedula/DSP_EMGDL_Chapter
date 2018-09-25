
from keras.layers import *
from keras import initializers, regularizers, constraints, optimizers
from keras.models import load_model, Model
import keras.backend as K

def AtzoriNet(input_shape, classes, n_pool='average', n_dropout=0., n_l2=0.0005, n_init='glorot_normal', batch_norm=False):
    """ Creates the Deep Neural Network architecture described in the paper of Manfredo Atzori:
    Deep Learning with Convolutional Neural Networks Applied to Electromyography Data: A Resource for the Classification of Movements for Prosthetic Hands
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5013051/
    
    Arguments:
        input_shape -- tuple, dimensions of the input in the form (height, width, channels)
        classes -- integer, number of classes to be classified, defines the dimension of the softmax unit
        n_pool -- string, pool method to be used {'max', 'average'}
        n_dropout -- float, rate of dropping units
        n_l2 -- float, ampunt of weight decay regularization
        n_init -- string, type of kernel initializer {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'normal', 'uniform'}
    
    Returns:
        model -- keras.models.Model (https://keras.io)
    """

    if n_init == 'glorot_normal':
        kernel_init = initializers.glorot_normal(seed=0)
    elif n_init == 'glorot_uniform':
        kernel_init = initializers.glorot_uniform(seed=0)
    elif n_init == 'he_normal':
        kernel_init = initializers.he_normal(seed=0)
    elif n_init == 'he_uniform':
        kernel_init = initializers.he_uniform(seed=0)
    elif n_init == 'normal':
        kernel_init = initializers.normal(seed=0)
    elif n_init == 'uniform':
        kernel_init = initializers.uniform(seed=0)
    # kernel_init = n_init
    kernel_regl = regularizers.l2(n_l2)

    ## Block 0 [Input]
    X_input = Input(input_shape, name='b0_input')
    X = X_input
    if batch_norm:
        X = BatchNormalization()(X)
    
    ## Block 1 [Pad -> Conv -> ReLU -> Dropout]
    X = Conv2D(32, (1, 10), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b1_conv2d_32_1x10')(X)
    X = Activation('relu', name='b1_relu')(X)
    X = Dropout(n_dropout, name='b1_dropout')(X)
    
    ## Block 2 [Pad -> Conv -> ReLU -> -> Dropout -> Pool]
    X = Conv2D(32, (3, 3), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b2_conv2d_32_3x3')(X)
    X = Activation('relu', name='b2_relu')(X)
    X = Dropout(n_dropout, name='b2_dropout')(X)
    if n_pool == 'max':
        X = MaxPooling2D((3,3), strides = (3,3), name='b2_pool')(X)
    else:
        X = AveragePooling2D((3,3), strides = (3,3), name='b2_pool')(X)
    
    ## Block 3 [Pad -> Conv -> ReLU -> Dropout -> Pool]
    X = Conv2D(64, (5, 5), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b3_conv2d_64_5x5')(X)
    X = Activation('relu', name='b3_relu')(X)
    X = Dropout(n_dropout, name='b3_dropout')(X)
    if n_pool == 'max':
        X = MaxPooling2D((3,3), strides = (3,3), name='b3_pool')(X)
    else:
        X = AveragePooling2D((3,3), strides = (3,3), name='b3_pool')(X)
    
    ## Block 4 [Pad -> Conv -> ReLU -> Dropout]
    X = Conv2D(64, (5, 1), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b4_conv2d_64_5x1')(X)
    X = Activation('relu', name='b4_relu')(X)
    X = Dropout(n_dropout, name='b4_dropout')(X)
    
    ## Block 5 [Pad -> Conv -> Softmax]
    X = Conv2D(classes, (1, 1), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b5_conv2d_{}_1x1'.format(classes))(X)
    X = Activation('softmax', name='b5_soft')(X)
    X = Reshape((-1,), name='b5_reshape')(X)
    
    model = Model(inputs = X_input, outputs = X, name='AtzoriNet')

    return model

def EmgLstmNet(input_shape, classes, n_dropout=0., n_l2=0.0005, n_init='glorot_normal', lstm_units=[256]):
    
    if n_init == 'glorot_normal':
        kernel_init = initializers.glorot_normal(seed=0)
    elif n_init == 'glorot_uniform':
        kernel_init = initializers.glorot_uniform(seed=0)
    elif n_init == 'he_normal':
        kernel_init = initializers.he_normal(seed=0)
    elif n_init == 'he_uniform':
        kernel_init = initializers.he_uniform(seed=0)
    elif n_init == 'normal':
        kernel_init = initializers.normal(seed=0)
    elif n_init == 'uniform':
        kernel_init = initializers.uniform(seed=0)
    kernel_regl = regularizers.l2(n_l2)

    x_input = Input(input_shape)
    x = Masking(-10.0)(x_input)

    for i in range(len(lstm_units)-1):
        x = LSTM(lstm_units[i],
        dropout=n_dropout, recurrent_dropout=n_dropout,
        kernel_regularizer=kernel_regl, kernel_initializer=kernel_init,
        recurrent_regularizer=kernel_regl, recurrent_initializer=kernel_init,
        return_sequences=True, input_shape=input_shape)(x)

    x = LSTM(lstm_units[-1],
        dropout=n_dropout, recurrent_dropout=n_dropout,
        kernel_regularizer=kernel_regl, kernel_initializer=kernel_init,
        recurrent_regularizer=kernel_regl, recurrent_initializer=kernel_init,
        return_sequences=False)(x)

    y = Dense(classes, activation='softmax', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init)(x)

    model = Model(x_input, y)
    return model


def GengNet(input_shape, classes, n_dropout=0.5, n_l2=0.0005, n_init='glorot_normal'):
    if n_init == 'glorot_normal':
        kernel_init = initializers.glorot_normal(seed=0)
    elif n_init == 'glorot_uniform':
        kernel_init = initializers.glorot_uniform(seed=0)
    elif n_init == 'he_normal':
        kernel_init = initializers.he_normal(seed=0)
    elif n_init == 'he_uniform':
        kernel_init = initializers.he_uniform(seed=0)
    elif n_init == 'normal':
        kernel_init = initializers.normal(seed=0)
    elif n_init == 'uniform':
        kernel_init = initializers.uniform(seed=0)

    kernel_regl = regularizers.l2(n_l2)

    input_img = Input((input_shape))
    x = BatchNormalization()(input_img)

    x = Conv2D(64, (3, 3), kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = LocallyConnected2D(64, (1, 1), kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = LocallyConnected2D(64, (1, 1), kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(n_dropout)(x)

    x = Dense(512, kernel_regularizer=kernel_regl, kernel_initializer=kernel_init)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(n_dropout)(x)

    x = Dense(512, kernel_regularizer=kernel_regl, kernel_initializer=kernel_init)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(n_dropout)(x)

    x = Dense(128, kernel_regularizer=kernel_regl, kernel_initializer=kernel_init)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(classes, kernel_regularizer=kernel_regl, kernel_initializer=kernel_init)(x)
    x = Activation('softmax')(x)

    model = Model(input_img, x, name='GengNet')
    return model


def getNetwork(network):

    if 'Atzori' in str(network):
        model = AtzoriNet
    elif 'EmgLstmNet' == str(network):
        model = EmgLstmNet
    elif 'Geng' in str(network):
        model = GengNet
    else:
        model = AtzoriNet
    return model



