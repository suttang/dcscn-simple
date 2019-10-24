from datetime import datetime
import math

import tensorflow as tf

import dataset


class Dcscn:
    def __init__(self):
        # Scale factor for Super Resolution (should be 2 or more)
        self.scale = 2

        # Number of feature extraction layers
        self.layers = 12

        # Number of image channels used. Now it should be 1. using only Y from YCbCr.
        self.input_channel = 1
        self.output_channel = 1

        # Number of filters of first feature-extraction CNNs
        self.filters = 196

        # Number of filters of last feature-extraction CNNs
        self.min_filters = 48

        # Number of CNN filters are decayed from [filters] to [min_filters] by this gamma
        self.filters_decay_gamma = 1.5

        # Initial weight stddev (won't be used when you use he or xavier initializer)
        self.weight_dev = 0.01

        # Output nodes should be kept by this probability. If 1, don't use dropout.
        self.dropout_rate = 0.8

        # Use batch normalization after each CNNs
        self.batch_norm = False

        # Norm for gradient clipping. If it's <= 0 we don't use gradient clipping.
        self.clipping_norm = 5

        # L2 decay
        self.l2_decay = 0.0001

        # Number of mini-batch images for training
        self.batch_size = 20

        self.learning_rate = 0.002

        self.H = []
        self.Weights = []
        self.Biases = []

    def _he_initializer(self, shape):
        n = shape[0] * shape[1] * shape[2]
        stddev = math.sqrt(2.0 / n)
        return tf.truncated_normal(shape=shape, stddev=stddev)

    def _weight(self, shape, name="weight"):
        initial = self._he_initializer(shape)
        return tf.Variable(initial, name=name)

    def _bias(self, shape, initial_value=0.0, name="bias"):
        initial = tf.constant(initial_value, shape=shape)
        return tf.Variable(initial, name=name)

    def _conv2d(self, input, w, stride, bias=None, use_batch_norm=False, name=""):
        output = tf.nn.conv2d(
            input,
            w,
            strides=[1, stride, stride, 1],
            padding="SAME",
            name=name + "_conv",
        )

        if bias is not None:
            output = tf.add(output, bias, name=name + "_add")

        if use_batch_norm:
            output = tf.layers.batch_normalization(
                output, training=self.is_training, name="BN"
            )

        return output

    def _prelu(self, input, features, name=""):
        with tf.variable_scope("prelu"):
            alphas = tf.Variable(
                tf.constant(0.1, shape=[features]), name=name + "_prelu"
            )

        output = tf.nn.relu(input) + tf.multiply(alphas, (input - tf.abs(input))) * 0.5
        return output

    def _convolutional_block(
        self,
        name,
        input,
        kernel_size,
        input_feature_num,
        output_feature_num,
        use_batch_norm=False,
        dropout_rate=1.0,
        dropout=None,
    ):
        with tf.variable_scope(name):
            shape_of_weight = [
                kernel_size,
                kernel_size,
                input_feature_num,
                output_feature_num,
            ]
            w = self._weight(shape=shape_of_weight, name="conv_W")

            shape_of_bias = [output_feature_num]
            b = self._bias(shape=shape_of_bias, name="conv_B")

            z = self._conv2d(
                input, w, stride=1, bias=b, use_batch_norm=use_batch_norm, name=name
            )

            if dropout_rate < 1.0:
                z = tf.nn.dropout(z, dropout, name="dropout")

            a = self._prelu(z, output_feature_num, name=name)

            self.H.append(a)

        self.Weights.append(w)
        self.Biases.append(b)

        return a

    def _pixel_shuffler(
        self, name, input, kernel_size, scale, input_feature_num, output_feature_num
    ):
        with tf.variable_scope(name):
            self._convolutional_block(
                name + "_CNN",
                input,
                kernel_size,
                input_feature_num=input_feature_num,
                output_feature_num=scale * scale * output_feature_num,
                use_batch_norm=False,
            )

            self.H.append(tf.depth_to_space(self.H[-1], scale))

    def placeholders(self, input_channel, output_channel):
        x = tf.placeholder(
            tf.float32, shape=[None, None, None, input_channel], name="x"
        )
        y = tf.placeholder(
            tf.float32, shape=[None, None, None, output_channel], name="y"
        )
        x2 = tf.placeholder(
            tf.float32, shape=[None, None, None, output_channel], name="x2"
        )
        learning_rate = tf.placeholder(tf.float32, shape=[], name="LearningRate")
        dropout = tf.placeholder(tf.float32, shape=[], name="dropout_keep_rate")
        is_training = tf.placeholder(tf.bool, name="is_training")

        return x, y, x2, learning_rate, dropout, is_training

    def forward(self, x, x2, dropout):
        # building feature extraction layers
        output_feature_num = self.filters
        total_output_feature_num = 0
        input_feature_num = self.input_channel
        input_tensor = x

        for i in range(self.layers):
            if self.min_filters != 0 and i > 0:
                x1 = i / float(self.layers - 1)
                y1 = pow(x1, 1.0 / self.filters_decay_gamma)
                output_feature_num = int(
                    (self.filters - self.min_filters) * (1 - y1) + self.min_filters
                )

                print(
                    "x1, {}, y1, {}, output_feature_num: {}".format(
                        x1, y1, output_feature_num
                    )
                )

            self._convolutional_block(
                "CNN%d" % (i + 1),
                input_tensor,
                kernel_size=3,
                input_feature_num=input_feature_num,
                output_feature_num=output_feature_num,
                use_batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate,
                dropout=dropout,
            )

            input_feature_num = output_feature_num
            input_tensor = self.H[-1]
            total_output_feature_num += output_feature_num

        with tf.variable_scope("Concat"):
            self.H_concat = tf.concat(self.H, 3, name="H_concat")

        # building reconstruction layers
        self._convolutional_block(
            "A1",
            self.H_concat,
            kernel_size=1,
            input_feature_num=total_output_feature_num,
            output_feature_num=64,
            dropout_rate=self.dropout_rate,
            dropout=dropout,
        )

        self._convolutional_block(
            "B1",
            self.H_concat,
            kernel_size=1,
            input_feature_num=total_output_feature_num,
            output_feature_num=32,
            dropout_rate=self.dropout_rate,
            dropout=dropout,
        )
        self._convolutional_block(
            "B2",
            self.H[-1],
            kernel_size=3,
            input_feature_num=32,
            output_feature_num=32,
            dropout_rate=self.dropout_rate,
            dropout=dropout,
        )
        self.H.append(tf.concat([self.H[-1], self.H[-3]], 3, name="Concat2"))

        # building upsampling layer
        pixel_shuffler_channel = 64 + 32
        self._pixel_shuffler(
            "Up-PS",
            self.H[-1],
            kernel_size=3,
            scale=self.scale,
            input_feature_num=pixel_shuffler_channel,
            output_feature_num=pixel_shuffler_channel,
        )

        self._convolutional_block(
            "R-CNN0",
            self.H[-1],
            kernel_size=3,
            input_feature_num=pixel_shuffler_channel,
            output_feature_num=self.output_channel,
        )

        y_hat = tf.add(self.H[-1], x2, name="output")

        return y_hat

    def loss(self, y_hat, y):
        diff = tf.subtract(y_hat, y, "diff")

        mse = tf.reduce_mean(tf.square(diff, name="diff_square"), name="mse")
        image_loss = tf.identity(mse, name="image_loss")

        l2_norm_losses = [tf.nn.l2_loss(w) for w in self.Weights]
        l2_norm_loss = self.l2_decay + tf.add_n(l2_norm_losses)
        loss = image_loss + l2_norm_loss

        tf.summary.scalar("loss", loss)

        return loss, image_loss, mse

    def optimizer(self, loss, learning_rate):
        beta1 = 0.9
        beta2 = 0.999
        if self.batch_norm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(
                    learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-8
                )
        else:
            optimizer = tf.train.AdamOptimizer(
                learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-8
            )

        trainables = tf.trainable_variables()
        grads = tf.gradients(loss, trainables)

        if self.clipping_norm > 0:
            clipped_grads, _ = tf.clip_by_global_norm(
                grads, clip_norm=self.clipping_norm
            )
            grad_var_pairs = zip(clipped_grads, trainables)
            training_optimizer = optimizer.apply_gradients(grad_var_pairs)
        else:
            training_optimizer = optimizer.minimize(loss)

        # Save weights
        for i in range(len(grads)):
            var = grads[i]
            mean_var = tf.reduce_mean(var)
            stddev_var = tf.sqrt(tf.reduce_mean(tf.square(var - mean_var)))
            tf.summary.scalar("{}/mean".format(var.name), var)
            tf.summary.scalar("{}/stddev".format(grads[i].name), stddev_var)
            tf.summary.histogram(grads[i].name, var)

        return training_optimizer

    def train(self, input, label):
        sess = tf.Session()

        log_dir = datetime.now().strftime("%Y%m%d%H%M%S")
        writer = tf.summary.FileWriter("logs/{}".format(log_dir), graph=sess.graph)

        x, y, x2, learning_rate, dropout, is_training = self.placeholders(
            input_channel=self.input_channel, output_channel=self.output_channel
        )

        # then you can use self.H_concat
        y_hat = self.forward(x, x2, dropout)
        loss, image_loss, mse = self.loss(y_hat, label)

        training = self.optimizer(loss, learning_rate)

        loader = dataset.Loader(
            "bsd200", scale=self.scale, image_size=48, batch_size=self.batch_size
        )

        sess.run(tf.global_variables_initializer())

        for i in range(100):
            input_images, upscaled_images, original_images = loader.feed()

            feed_dict = {
                x: input_images,
                x2: upscaled_images,
                y: original_images,
                learning_rate: self.learning_rate,
                dropout: self.dropout_rate,
                is_training: 1,
            }

            sess.run([training, loss, mse], feed_dict=feed_dict)

            print("Step: {}".format(i))

        writer.close()
        sess.close()
