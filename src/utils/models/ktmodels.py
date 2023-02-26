'''
    HyperModel definition for keras hyper-parameter tuning.

'''
# Dependencies ---------------------------------------------------------------
from keras_tuner import HyperModel
import tensorflow as tf
# ----------------------------------------------------------------------------


class CAE(HyperModel):
    """
    HyperModel class for the Convolutional AutoEncoder (CAE).

    Args:
        input_shape (tuple): Shape of the input images.
        bottleneck_filters (int, optional): Number of filters in the bottleneck
            layer. Defaults to 16.
    """

    def __init__(self, input_shape, bottleneck_filters: int = 16):
        self.input_shape = input_shape
        self.bottleneck_filters = bottleneck_filters

    def build(self, hp):

        # *** Initialize the Hyper-Parameters ---------------------------------
        self._init_hyperparameters(hp)

        # *** Encoder ---------------------------------------------------------
        input_img = tf.keras.layers.Input(shape=self.input_shape)

        regularizer = tf.keras.regularizers.l2(1e-5)
        conv2d_1 = tf.keras.layers.Conv2D(filters=self.filters_1,
                                          kernel_size=self.kernel_size_1,
                                          strides=2,
                                          activation='linear',
                                          padding='same',
                                          kernel_regularizer=regularizer
                                          )(input_img)

        con2d_act_1 = act_bnorm(conv2d_1)
        if self.double_1:
            conv2d_1 = tf.keras.layers.Conv2D(filters=self.filters_1,
                                              kernel_size=3,
                                              activation='linear',
                                              padding='same',
                                              kernel_regularizer=regularizer
                                              )(con2d_act_1)
            con2d_act_1 = act_bnorm(conv2d_1)

        regularizer = tf.keras.regularizers.l2(1e-5)
        conv2d_2 = tf.keras.layers.Conv2D(filters=self.filters_2,
                                          kernel_size=self.kernel_size_2,
                                          strides=2,
                                          padding='same',
                                          activation='linear',
                                          kernel_regularizer=regularizer
                                          )(con2d_act_1)
        con2d_act_2 = act_bnorm(conv2d_2)
        if self.double_2:
            conv2d_2 = tf.keras.layers.Conv2D(filters=self.filters_2,
                                              kernel_size=3,
                                              padding='same',
                                              activation='linear',
                                              kernel_regularizer=regularizer
                                              )(con2d_act_2)
            con2d_act_2 = act_bnorm(conv2d_2)

        regularizer = tf.keras.regularizers.l2(1e-5)
        conv2d_3 = tf.keras.layers.Conv2D(filters=self.filters_3,
                                          kernel_size=self.kernel_size_3,
                                          strides=2,
                                          padding='same',
                                          activation='linear',
                                          kernel_regularizer=regularizer
                                          )(con2d_act_2)
        con2d_act_3 = act_bnorm(conv2d_3)

        if self.double_3:
            conv2d_3 = tf.keras.layers.Conv2D(filters=self.filters_3,
                                              kernel_size=3,
                                              padding='same',
                                              activation='linear',
                                              kernel_regularizer=regularizer
                                              )(con2d_act_3)
            con2d_act_3 = act_bnorm(conv2d_3)

        conv2d_4 = tf.keras.layers.Conv2D(filters=self.filters_4,
                                          kernel_size=self.kernel_size_4,
                                          padding='same',
                                          activation='linear',
                                          )(con2d_act_3)
        bottleneck = act_bnorm(conv2d_4, name='bottleneck')

        # * Start Decoder -----------------------------------------------------

        conv2d_4 = tf.keras.layers.Conv2D(filters=self.filters_4,
                                          kernel_size=self.kernel_size_4,
                                          padding='same',
                                          activation='linear'
                                          )(bottleneck)
        con2d_act_4 = act_bnorm(conv2d_4)

        if self.double_3:
            conv2d_3 = tf.keras.layers.Conv2D(filters=self.filters_3,
                                              kernel_size=3,
                                              padding='same',
                                              activation='linear',
                                              kernel_regularizer=regularizer
                                              )(con2d_act_4)
            con2d_act_4 = act_bnorm(conv2d_3)

        upsample_3 = tf.keras.layers.UpSampling2D(2)(con2d_act_4)
        regularizer = tf.keras.regularizers.l2(1e-5)
        conv2d_3 = tf.keras.layers.Conv2D(filters=self.filters_3,
                                          kernel_size=self.kernel_size_3,
                                          padding='same',
                                          activation='linear',
                                          kernel_regularizer=regularizer
                                          )(upsample_3)
        con2d_act_3 = act_bnorm(conv2d_3)

        if self.double_2:
            conv2d_2 = tf.keras.layers.Conv2D(filters=self.filters_2,
                                              kernel_size=3,
                                              padding='same',
                                              activation='linear',
                                              kernel_regularizer=regularizer
                                              )(con2d_act_3)
            con2d_act_3 = act_bnorm(conv2d_2)

        upsample_2 = tf.keras.layers.UpSampling2D(2)(con2d_act_3)
        regularizer = tf.keras.regularizers.l2(1e-5)
        conv2d_2 = tf.keras.layers.Conv2D(filters=self.filters_2,
                                          kernel_size=self.kernel_size_2,
                                          padding='same',
                                          activation='linear',
                                          kernel_regularizer=regularizer
                                          )(upsample_2)
        con2d_act_2 = act_bnorm(conv2d_2)

        if self.double_1:
            conv2d_1 = tf.keras.layers.Conv2D(filters=self.filters_1,
                                              kernel_size=3,
                                              padding='same',
                                              activation='linear',
                                              kernel_regularizer=regularizer
                                              )(con2d_act_2)
            con2d_act_2 = act_bnorm(conv2d_1)

        upsample_1 = tf.keras.layers.UpSampling2D(2)(con2d_act_2)
        regularizer = tf.keras.regularizers.l2(1e-5)
        conv2d_1 = tf.keras.layers.Conv2D(filters=self.filters_1,
                                          kernel_size=self.kernel_size_1,
                                          padding='same',
                                          activation='linear',
                                          kernel_regularizer=regularizer
                                          )(upsample_1)
        con2d_act_1 = act_bnorm(conv2d_1)

        final_conv = tf.keras.layers.Conv2D(
            filters=self.input_shape[2],
            kernel_size=1,
            padding='same',
            activation='sigmoid')(con2d_act_1)
        model = tf.keras.models.Model(input_img, final_conv)

        model.compile(loss=['mse'],
                      optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate),
            metrics=['mae', 'mse']
        )

        return model

    def _init_hyperparameters(self, hp):
        # First convolutional block
        self.filters_1 = hp.Int(
            "filters1", min_value=32, max_value=96, step=32)
        self.kernel_size_1 = hp.Int(
            "kernel_size1", min_value=2, max_value=6, step=2)

        # Second convolutional block
        self.filters_2 = hp.Int(
            "filters2", min_value=64, max_value=224, step=32)
        self.kernel_size_2 = hp.Int(
            "kernel_size2", min_value=2, max_value=10, step=2)

        # Third convolutional block
        self.filters_3 = hp.Int(
            "filters3", min_value=224, max_value=288, step=32)
        self.kernel_size_3 = hp.Int(
            "kernel_size3", min_value=2, max_value=6, step=2)
        # Fourth convolutional block
        self.filters_4 = self.bottleneck_filters
        self.kernel_size_4 = hp.Int(
            "kernel_size4", min_value=2, max_value=9, step=1)

        # Optimization
        self.learning_rate = hp.Choice(
            "learning_rate", values=[5e-4, 7.5e-4, 1e-3])

        # Depth
        self.double_1 = hp.Choice('double1', values=[True, False])
        self.double_2 = hp.Choice('double2', values=[True, False])
        self.double_3 = hp.Choice('double3', values=[True, False])


def act_bnorm(inputs: tf.Tensor, activation: str = 'relu',
              name: str = '') -> tf.Tensor:
    """
    Activation and Batch Normalization

    Args:
        inputs (tf.Tensor): Input tensor
        activation (str, optional): Activation function. Defaults to 'relu'.
        name (str, optional): Name of the activation layer. Defaults to ''.

    Returns:
        tf.Tensor: Output tensor
    """
    bn = tf.keras.layers.BatchNormalization()(inputs)
    if name:
        act = tf.keras.layers.Activation(activation, name=name)(bn)
    else:
        act = tf.keras.layers.Activation(activation)(bn)

    return act
