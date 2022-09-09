import tensorflow as tf
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Conv1D, Activation, Dense, \
                                    concatenate, BatchNormalization, GlobalAveragePooling1D, \
                                    Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, \
                                    ReLU, MaxPooling2D, Flatten, Dropout, LSTM, Reshape

'''
def CNN_2D_2D_model(image_length=224, training=False):
    ################# CNN stft ################################
    input_stft = Input(shape=(image_length, image_length, 1))
    base_model_stft = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                        input_shape=(image_length, image_length, 1),
                                                        weights=None)
    model_stft = base_model_stft(input_stft, training=training)
    output_stft = Model(input_stft, model_stft)
    output_stft = output_stft([input_stft])
    output_stft = GlobalAveragePooling2D()(output_stft)
    output_stft = BatchNormalization()(output_stft, training=training)
    output_stft = tf.keras.layers.Dense(1024,  activation='relu',
                                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4),
                                  activity_regularizer=regularizers.l2(1e-5))(output_stft)

    ################# CNN mel ################################
    input_mel = Input(shape=(image_length, image_length, 1))
    base_model_mel = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                            input_shape=(image_length, image_length, 1),
                                                            weights=None)
    model_mel = base_model_mel(input_mel, training=training)
    output_mel = Model(input_mel, model_mel)
    output_mel = output_mel([input_mel])
    output_mel = GlobalAveragePooling2D()(output_mel)
    output_mel = BatchNormalization()(output_mel, training=training)
    output_mel = tf.keras.layers.Dense(1024,  activation='relu',
                                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4),
                                  activity_regularizer=regularizers.l2(1e-5))(output_mel)

    ################# CNN mel vs stft ################################
    output = output_stft + output_mel
    output = Activation('relu')(output)
    output = Dropout(0.2)(output, training=training)
    output = Dense(4, activation='softmax', 
                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            bias_regularizer=regularizers.l2(1e-4),
                            activity_regularizer=regularizers.l2(1e-5))(output)

    network = Model(inputs=[input_mel, input_stft], outputs=output)
    return network
'''


def CNN_2D_2D_model(image_length=224, training=False):
    ################# CNN stft ################################
    input_stft = Input(shape=(image_length, image_length, 1))
    base_model_stft = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                        input_shape=(image_length, image_length, 1),
                                                        weights=None)
    model_stft = base_model_stft(input_stft, training=training)
    output_stft = Model(input_stft, model_stft)
    output_stft = output_stft([input_stft])
    output_stft = tf.keras.layers.Dense(output_stft.shape[-1],  activation='relu',
                                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4),
                                  activity_regularizer=regularizers.l2(1e-5))(output_stft)
    output_stft  = MultiHeadAttention(head_size=16, num_heads=16)([output_stft, output_stft, output_stft]) 
    output_stft = GlobalAveragePooling2D()(output_stft)
    output_stft = BatchNormalization()(output_stft, training=training)
    output_stft = tf.keras.layers.Dense(1024,  activation='relu',
                                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4),
                                  activity_regularizer=regularizers.l2(1e-5))(output_stft)

    ################# CNN mel ################################
    input_mel = Input(shape=(image_length, image_length, 1))
    base_model_mel = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                            input_shape=(image_length, image_length, 1),
                                                            weights=None)
    model_mel = base_model_mel(input_mel, training=training)
    output_mel = Model(input_mel, model_mel)
    output_mel = output_mel([input_mel])
    output_mel = tf.keras.layers.Dense(output_mel.shape[-1],  activation='relu',
                                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4),
                                  activity_regularizer=regularizers.l2(1e-5))(output_mel)
#     output_mel = Dropout(0.2)(output_mel, training=training)
    output_mel  = MultiHeadAttention(head_size=16, num_heads=16)([output_mel, output_mel, output_mel]) 
    output_mel = GlobalAveragePooling2D()(output_mel)
    output_mel = BatchNormalization()(output_mel, training=training)
    output_mel = tf.keras.layers.Dense(1024,  activation='relu',
                                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                  bias_regularizer=regularizers.l2(1e-4),
                                  activity_regularizer=regularizers.l2(1e-5))(output_mel)

    ################# CNN mel vs stft ################################
#     output = concatenate((output_mel, output_stft))
    output = output_stft + output_mel
    output = Activation('relu')(output)
    output = Dropout(0.2)(output, training=training)
    output = Dense(4, activation='softmax', 
                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                            bias_regularizer=regularizers.l2(1e-4),
                            activity_regularizer=regularizers.l2(1e-5))(output)

    network = Model(inputs=[input_mel, input_stft], outputs=output)
    return network
