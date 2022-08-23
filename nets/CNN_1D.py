from functools import partial
import keras
import tensorflow as tf
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.layers import LSTM, Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout
import keras.backend as K
from keras import layers, regularizers
from keras.models import Model
from tensorflow.keras.applications import DenseNet121


def TransformerLayer(x, c, num_heads=4, training=None):
    x = tf.keras.layers.Dense(c,  activation='relu',
                                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4),
                                  activity_regularizer=regularizers.l2(1e-5))(x)
#     x = Dropout(0.1)(x, training=training)
    ma  = MultiHeadAttention(head_size=num_heads, num_heads=num_heads)([x, x, x]) 
    ma = BatchNormalization()(ma, training=training)
    ma = tf.keras.layers.Dense(c,   activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(ma) 
    ma = tf.keras.layers.Dense(c,  activation='relu',
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(ma) 
    ma = Dropout(0.1)(ma, training=training)
    ma = tf.keras.layers.GRU(c, return_sequences=False)(ma)
    return ma

# For m34 Residual, use RepeatVector. Or tensorflow backend.repeat
def identity_block(input_tensor, kernel_size, filters, stage, block, training):
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5),
              name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=training)
    x = Activation('relu')(x)

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5),
              name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=training)

    if input_tensor.shape[2] != x.shape[2]:
        x = layers.add([x, Lambda(lambda y: K.repeat_elements(y, rep=2, axis=2))(input_tensor)])
    else:
        x = layers.add([x, input_tensor])

    x = BatchNormalization()(x, training=training)
    x = Activation('relu')(x)
    return x
  
def cnn_1d_model(input_shape, training=None):
    '''
    The model was rebuilt based on the construction of resnet 34 and inherited from this source code:
    https://github.com/philipperemy/very-deep-convnets-raw-waveforms/blob/master/model_resnet.py
    '''
    inputs = Input(shape=[input_shape, 1])
    x = Conv1D(64,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=4, strides=None)(x)


    for i in range(3):
        x = identity_block(x, kernel_size=3, filters=64, stage=1, block=i, training=training)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(4):
        x = identity_block(x, kernel_size=3, filters=128, stage=2, block=i, training=training)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(23):
        x = identity_block(x, kernel_size=3, filters=256, stage=3, block=i, training=training)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(3):
        x = identity_block(x, kernel_size=3, filters=512, stage=4, block=i, training=training)
    x = Dense(256, activation=ReLU(), 
                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            bias_regularizer=regularizers.l2(1e-4),
                            activity_regularizer=regularizers.l2(1e-5))(x)
    x = tf.keras.layers.Bidirectional(LSTM(units=256, return_sequences=False, activation='relu'))(x)
    x = Dropout(0.1)(x, training=training)
    # x = TransformerLayer(x, 256, num_heads=4, training=None)
    m_1 = Model(inputs, x)
    return m_1
