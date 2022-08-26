from nets.CNN_1D import cnn_1d_model, TransformerLayer
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Conv1D, Activation, Dense, \
                                    concatenate, BatchNormalization, GlobalAveragePooling1D, \
                                    Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, \
                                    ReLU, MaxPooling2D, Flatten, Dropout, LSTM, Reshape


def CNN_1D_2D_model(image_length=224, fft_length=64653, training=False):
    ################# CNN 2D ################################
    input_2D = Input(shape=(image_length, image_length, 1))
    base_model_2D = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                            input_shape=(image_length, image_length, 1),
                                                            weights=None)
    model_2D = base_model_2D(input_2D, training=training)
    output_2D = Model(input_2D, model_2D)
    output_2D = output_2D([input_2D])

    # output_2D = Reshape((output_2D.shape[-2]*output_2D.shape[-3], output_2D.shape[-1]))(output_2D)
    # output_2D = TransformerLayer(output_2D, output_2D.shape[-1], num_heads=8, training=training)

    output_2D = GlobalAveragePooling2D()(output_2D)
    output_2D = Dense(1024, activation=ReLU(), 
                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            bias_regularizer=regularizers.l2(1e-4),
                            activity_regularizer=regularizers.l2(1e-5))(output_2D)
    output_2D = Dropout(0.1)(output_2D, training=training)

    ################# CNN 1D ################################
    input_1D = Input(shape=(fft_length, 1))
    base_model_1D = cnn_1d_model(fft_length, )
    output_1D = base_model_1D([input_1D])
#     output_1D = Dense(256, activation=ReLU(), 
#                             kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                             bias_regularizer=regularizers.l2(1e-4),
#                             activity_regularizer=regularizers.l2(1e-5))(output_1D)
#     output_1D = Dropout(0.1)(output_1D, training=training)

    ################# CNN 1D vs 2D ################################
    output = concatenate((output_1D, output_2D))
    output = Dense(4, activation='softmax', 
                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            bias_regularizer=regularizers.l2(1e-4),
                            activity_regularizer=regularizers.l2(1e-5))(output)

    network = Model(inputs=[input_2D, input_1D], outputs=output)
    return network
