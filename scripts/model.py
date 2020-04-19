"""Return the model"""
import tensorflow as tf

import data


class Conv2D_BN:
    def __init__(self, filters, kernel_size=(3, 3), activation='relu', padding='valid', strides=(1, 1)):
        self.filters = int(filters)
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.strides = strides

    def __call__(self, model):
        x = tf.keras.layers.Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding, strides=self.strides)(model)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(self.activation)(x)
        return x


class UpConv2D_BN:
    def __init__(self, filters, kernel_size=(2, 2), activation='relu', padding='valid', strides=2):
        self.filters = int(filters)
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.strides = strides

    def __call__(self, model):
        x = tf.keras.layers.Conv2DTranspose(self.filters,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            strides=self.strides)(model)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(self.activation)(x)
        return x


@tf.function
def dsc(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) +
                                            tf.reduce_sum(y_pred_f) +
                                            smooth)


@tf.function
def dice_loss(y_true, y_pred):
    return 1 - dsc(y_true, y_pred)


@tf.function
def tversky(y_true, y_pred):
    alpha = 0.7
    smooth = 1.0
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


@tf.function
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


@tf.function
def focal_tversky(y_true, y_pred):
    return tf.pow((1 - tversky(y_true, y_pred)), 0.75)


@tf.function
def iou(y_true, y_pred):
    intersect = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    union = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    return tf.reduce_mean(tf.math.divide_no_nan(intersect, (union - intersect)), axis=1)


@tf.function
def mean_iou(y_true, y_pred):
    y_true_32 = tf.cast(y_true, tf.float32)
    y_pred_32 = tf.cast(y_pred, tf.float32)
    score = tf.map_fn(lambda x: iou(y_true_32, tf.cast(y_pred_32 > x, tf.float32)),
                      tf.range(0.5, 1.0, 0.05, tf.float32),
                      tf.float32)
    return tf.reduce_mean(score)


@tf.function
def iou_loss(y_true, y_pred):
    return -1*mean_iou(y_true, y_pred)


def get_callbacks(name):
    return [
        tf.keras.callbacks.ModelCheckpoint(f'models/{name}_all.h5',
                                           save_best_only=False,
                                           save_weights_only=True,
                                           verbose=0),
        tf.keras.callbacks.ModelCheckpoint(f'models/{name}_best.h5',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1)
    ]


def get_do_unet_scale_invariant():
    np_filters = 32

    inputs = tf.keras.layers.Input((380, 380, 3))
    down0 = Conv2D_BN(3*np_filters, kernel_size=(3, 3))(inputs) # 378x378
    down0 = Conv2D_BN(np_filters, kernel_size=(1, 1))(down0)  # 378x378
    down0 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down0)  # 376x376
    pool0 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down0)  # 188x188

    np_filters *= 2
    down1 = Conv2D_BN(np_filters, kernel_size=(3, 3))(pool0) # 186x186
    down1 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down1)  # 184x184
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down1)  # 92x92

    np_filters *= 2
    down2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(pool1)  # 90x90
    down2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down2)  # 88x88
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down2)  # 44x44

    np_filters *= 2
    down3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(pool2)  # 42x42
    down3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down3)  # 40x40
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down3)  # 20x20

    np_filters *= 2
    down4 = Conv2D_BN(np_filters, kernel_size=(3, 3))(pool3)  # 18x18
    down4 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down4)  # 16x16

    np_filters /= 2
    up1 = UpConv2D_BN(np_filters)(down4)  # 32x32
    up1 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(4)(down3), up1])  # 32x32
    up1 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up1)  # 30x30
    up1 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up1)  # 28x28

    np_filters /= 2
    up2 = UpConv2D_BN(np_filters)(up1)  #56x56
    up2 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(16)(down2), up2])  # 56x56
    up2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up2)  # 54x54
    up2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up2)  # 52x52

    np_filters /= 2
    up3 = UpConv2D_BN(np_filters)(up2)  # 104x104
    up3 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(40)(down1), up3])  # 104x104
    up3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up3)  # 102x102
    up3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up3)  # 100x100

    np_filters /= 2
    up4 = UpConv2D_BN(np_filters)(up3)  # 200x200
    up4 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(88)(down0), up4])  # 200x200
    up4 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up4)  # 198x198
    up4 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up4)  # 196x196

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='softmax', name='mask')(up4)
    out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='softmax', name='edge')(up4)

    model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))

    opt = tf.optimizers.Adam(lr=0.0001)

    model.compile(loss={'mask': tversky_loss,
                        'edge': tversky_loss},
                  loss_weights=[0.3, 0.7],
                  optimizer=opt,
                  metrics={'mask': [mean_iou, dsc, tversky], 
                           'edge': [mean_iou, dsc, tversky]})

    return model


def get_do_unet():
    np_filters = 32

    inputs = tf.keras.layers.Input((188, 188, 3))
    down1 = Conv2D_BN(3*np_filters, kernel_size=(3, 3))(inputs) # 186x186
    down1 = Conv2D_BN(np_filters, kernel_size=(1, 1))(down1)  # 186x186
    down1 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down1)  # 184x184
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down1)  # 92x92

    np_filters *= 2
    down2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(pool1)  # 90x90
    down2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down2)  # 88x88
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down2)  # 44x44

    np_filters *= 2
    down3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(pool2)  # 42x42
    down3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down3)  # 40x40
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(down3)  # 20x20

    np_filters *= 2
    down4 = Conv2D_BN(np_filters, kernel_size=(3, 3))(pool3)  # 18x18
    down4 = Conv2D_BN(np_filters, kernel_size=(3, 3))(down4)  # 16x16

    np_filters /= 2
    up1 = UpConv2D_BN(np_filters)(down4)  # 32x32
    up1 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(4)(down3), up1])  # 32x32
    up1 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up1)  # 30x30
    up1 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up1)  # 28x28

    np_filters /= 2
    up2 = UpConv2D_BN(np_filters)(up1)  #56x56
    up2 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(16)(down2), up2])  # 56x56
    up2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up2)  # 54x54
    up2 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up2)  # 52x52

    np_filters /= 2
    up3 = UpConv2D_BN(np_filters)(up2)  # 104x104
    up3 = tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(40)(down1), up3])  # 104x104
    up3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up3)  # 102x102
    up3 = Conv2D_BN(np_filters, kernel_size=(3, 3))(up3)  # 100x100

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='softmax', name='mask')(up3)
    out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='softmax', name='edge')(up3)

    model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))

    opt = tf.optimizers.Adam(lr=0.0001)

    model.compile(loss={'mask': tversky_loss,
                        'edge': tversky_loss},
                  loss_weights=[0.3, 0.7],
                  optimizer=opt,
                  metrics={'mask': [mean_iou, dsc, tversky], 
                           'edge': [mean_iou, dsc, tversky]})

    return model


class DO_UNet:
    def __init__(self, train_files, test_files, scale_invariant=True):
        self.scale_invariant = scale_invariant

        self.train_dataset = self.generate_train_dataset(train_files)
        self.test_dataset = self.generate_test_dataset(test_files)

        if self.scale_invariant:
            self.model = get_do_unet_scale_invariant()
        else:
            self.model = get_do_unet()

    def generate_train_dataset(self, img_files):
        imgs, mask, edge = data.load_data(img_files)

        def train_gen():
            if self.scale_invariant:
                return data.train_generator(imgs, mask,
                                            edge=edge,
                                            padding=200,
                                            input_size=380,
                                            output_size=196,
                                            scale_range=0.5)

            return data.train_generator(imgs, mask,
                                        edge=edge,
                                        padding=100,
                                        input_size=188,
                                        output_size=100)

        if self.scale_invariant:
            return tf.data.Dataset.from_generator(train_gen,
                                                  (tf.float64, ((tf.float64), (tf.float64))),
                                                  ((380, 380, 3), ((196, 196, 1), (196, 196, 1)))
                                                 )

        return tf.data.Dataset.from_generator(train_gen,
                                              (tf.float64, ((tf.float64), (tf.float64))),
                                              ((188, 188, 3), ((100, 100, 1), (100, 100, 1)))
                                             )


    def generate_test_dataset(self, img_files):
        imgs, mask, edge = data.load_data(img_files)

        if self.scale_invariant:
            img_chips, mask_chips, edge_chips = data.test_chips(imgs, mask,
                                                                edge=edge,
                                                                padding=200,
                                                                input_size=380,
                                                                output_size=196)
        else:
            img_chips, mask_chips, edge_chips = data.test_chips(imgs, mask,
                                                                edge=edge,
                                                                padding=100,
                                                                input_size=188,
                                                                output_size=100)

        return tf.data.Dataset.from_tensor_slices((img_chips,
                                                   (mask_chips, edge_chips))
                                                 )

    def compile(self, optimiser='adam', learning_rate=0.001):
        if optimiser == 'adam':
            opt = tf.optimizers.Adam(lr=learning_rate)
        elif optimiser == 'sgd':
            opt = tf.optimizers.SGD(lr=learning_rate)

            self.model.compile(loss={'mask': tversky_loss,
                                     'edge': tversky_loss},
                               loss_weights=[0.3, 0.7],
                               optimizer=opt,
                               metrics={'mask': [mean_iou, dsc, tversky], 
                                        'edge': [mean_iou, dsc, tversky]})

    def fit(self, model_name,
            epochs=40,
            imgs_per_epoch=1000,
            batchsize=8,
            workers=8):

        return self.model.fit(self.train_dataset.batch(batchsize),
                              epochs=epochs,
                              steps_per_epoch=(imgs_per_epoch // batchsize),
                              validation_data=self.test_dataset.batch(batchsize),
                              max_queue_size=2*workers,
                              use_multiprocessing=False,
                              workers=8,
                              verbose=1,
                              callbacks=get_callbacks(model_name))








