from datetime import datetime
import os
import math

import tensorflow as tf

import dataset
from utils import add_summaries
from utils import align_image
from utils import convert_rgb_to_y
from utils import convert_rgb_to_ycbcr
from utils import convert_y_and_cbcr_to_rgb
from utils import resize_image
from utils import save_image
from utils import load_image
from utils import calc_psnr_and_ssim
from utils import get_validation_files


class Dcscn:
    def __init__(self, with_restore=False):
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

        # Number of CNN filters are decayed
        # from [filters] to [min_filters] by this gamma
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
        self.l2_decay = 0.00003

        # Number of mini-batch images for training
        self.batch_size = 5

        self.initial_learning_rate = 0.0002

        self.H = []
        self.Weights = []
        self.Biases = []

        # Restore model path
        self.is_use_restore = False

        if with_restore:
            self.is_use_restore = True
            self.restore_model_path = with_restore

        # Build graph
        x, y, x2, learning_rate, dropout, is_training = self.placeholders(
            input_channel=self.input_channel, output_channel=self.output_channel
        )
        self.x = x
        self.x2 = x2
        self.y = y
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.is_training = is_training
        self.y_hat = self.forward(self.x, self.x2, self.dropout)

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

            a = self._prelu(z, output_feature_num, name=name)

            if dropout_rate < 1.0:
                a = tf.nn.dropout(a, dropout, name="dropout")

            self.H.append(a)

            add_summaries("weight", name, w, save_stddev=True, save_mean=True)
            add_summaries("output", name, a, save_stddev=True, save_mean=True)
            add_summaries("bias", name, b, save_stddev=True, save_mean=True)

            # # Save image
            # shapes = w.get_shape().as_list()
            # weights = tf.reshape(w, [shapes[0], shapes[1], shapes[2] * shapes[3]])
            # weights_transposed = tf.transpose(weights, [2, 0, 1])
            # weights_transposed = tf.reshape(
            #     weights_transposed, [shapes[2] * shapes[3], shapes[0], shapes[1], 1]
            # )
            # tf.summary.image("weights", weights_transposed, max_outputs=6)

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

    def _calc_filters(self, first, last, layers, decay):
        return [
            int((first - last) * (1 - pow(i / float(layers - 1), 1.0 / decay)) + last)
            for i in range(layers)
        ]

    def forward(self, input, x2, dropout):
        # building feature extraction layers
        total_output_feature_num = 0

        with tf.name_scope("X_"):
            mean_var = tf.reduce_mean(input)
            stddev_var = tf.sqrt(tf.reduce_mean(tf.square(input - mean_var)))
            tf.summary.scalar("output/mean", mean_var)
            tf.summary.scalar("output/stddev", stddev_var)

        filters = self._calc_filters(
            self.filters, self.min_filters, self.layers, self.filters_decay_gamma
        )

        input_filter = self.input_channel
        for i, filter in enumerate(filters):
            self._convolutional_block(
                "CNN%d" % (i + 1),
                input,
                kernel_size=3,
                input_feature_num=input_filter,
                output_feature_num=filter,
                use_batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate,
                dropout=dropout,
            )
            input_filter = filter
            input = self.H[-1]
            total_output_feature_num += filter

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

        with tf.name_scope("Y_"):
            mean = tf.reduce_mean(y_hat)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(y_hat - mean)))
            tf.summary.scalar("output/mean", mean)
            tf.summary.scalar("output/stddev", stddev)
            tf.summary.histogram("output", y_hat)

        return y_hat

    def loss(self, y_hat, y):
        diff = tf.subtract(y_hat, y, "diff")

        mse = tf.reduce_mean(tf.square(diff, name="diff_square"), name="mse")
        image_loss = tf.identity(mse, name="image_loss")

        l2_norm_losses = [tf.nn.l2_loss(w) for w in self.Weights]
        l2_norm_loss = self.l2_decay + tf.add_n(l2_norm_losses)
        loss = image_loss + l2_norm_loss

        tf.summary.scalar("Loss", loss)
        tf.summary.scalar("L2WeightDecayLoss", l2_norm_loss)

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
        # for i in range(len(grads)):
        #     var = grads[i]
        #     mean_var = tf.reduce_mean(var)
        #     stddev_var = tf.sqrt(tf.reduce_mean(tf.square(var - mean_var)))

        #     # tf.summary.scalar("{}/mean".format(var.name), var)
        #     # tf.summary.scalar("{}/stddev".format(grads[i].name), stddev_var)
        #     # tf.summary.histogram(grads[i].name, var)

        return training_optimizer

    def train(self, output_path, validation_dataset=None):
        loss, image_loss, mse = self.loss(self.y_hat, self.y)

        training = self.optimizer(loss, self.learning_rate)

        loader = dataset.Loader(
            "bsd200", scale=self.scale, image_size=48, batch_size=self.batch_size
        )

        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            log_dir = datetime.now().strftime("%Y%m%d%H%M%S")
            writer = tf.summary.FileWriter("logs/{}".format(log_dir), graph=sess.graph)

            sess.run(tf.global_variables_initializer())

            for i in range(8000 * 100):
                input_images, upscaled_images, original_images = loader.feed()

                feed_dict = {
                    self.x: input_images,
                    self.x2: upscaled_images,
                    self.y: original_images,
                    self.learning_rate: self.initial_learning_rate,
                    self.dropout: self.dropout_rate,
                    self.is_training: 1,
                }

                _, s_loss, s_mse = sess.run([training, loss, mse], feed_dict=feed_dict)
                print("Step: {}, loss: {}, mse: {}".format(i, s_loss, s_mse))

                if i % 100 == 0:
                    summarized, _ = sess.run([summary, loss], feed_dict=feed_dict)
                    writer.add_summary(summarized, i)

                    # Learning rate
                    lr_summary = tf.Summary(
                        value=[
                            tf.Summary.Value(
                                tag="LR", simple_value=self.initial_learning_rate
                            )
                        ]
                    )
                    writer.add_summary(lr_summary, i)

                if i % 8000 == 0:
                    # Metrics
                    if validation_dataset is not None:
                        validation_files = get_validation_files(validation_dataset)
                        psnr, ssim = self.calc_metrics(validation_files)
                        print("PSNR: {}, SSSIM: {}".format(psnr, ssim))
                        psnr_summary = tf.Summary(
                            value=[tf.Summary.Value(tag="PSNR", simple_value=psnr)]
                        )
                        ssim_summary = tf.Summary(
                            value=[tf.Summary.Value(tag="SSIM", simple_value=ssim)]
                        )
                        writer.add_summary(psnr_summary, i)
                        writer.add_summary(ssim_summary, i)

            writer.close()

            # Save model
            output_dir = os.path.dirname(os.path.join(os.getcwd(), output_path))
            os.makedirs(output_dir, exist_ok=True)
            self.save(sess, output_path)

    def run(self, input_image, input_bicubic_image):
        h, w = input_image.shape[:2]
        ch = input_image.shape[2] if len(input_image.shape) > 2 else 1

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Restore model
            if self.is_use_restore:
                restore_path = os.path.join(
                    os.getcwd(), self.restore_model_path + ".ckpt"
                )
                saver = tf.train.Saver()
                saver.restore(sess, restore_path)

            feed_dict = {
                self.x: input_image.reshape(1, h, w, ch),
                self.x2: input_bicubic_image.reshape(
                    1,
                    self.scale * input_image.shape[0],
                    self.scale * input_image.shape[1],
                    ch,
                ),
                self.learning_rate: self.initial_learning_rate,
                self.dropout: 1.0,
                self.is_training: 0,
            }
            y_hat = sess.run([self.y_hat], feed_dict=feed_dict)
            output = y_hat[0][0]

        return output

    def inference(self, input_image, output_dir, save_images=False):
        # Create scaled image
        scaled_image = resize_image(input_image, 2)

        # Create y and scaled y image
        input_y_image = convert_rgb_to_y(input_image)
        scaled_y_image = resize_image(input_y_image, self.scale)

        output_y_image = self.run(input_y_image, scaled_y_image)

        # Create result image
        scaled_ycbcr_image = convert_rgb_to_ycbcr(scaled_image)
        result_image = convert_y_and_cbcr_to_rgb(
            output_y_image, scaled_ycbcr_image[:, :, 1:3]
        )

        if save_images:
            save_image(input_image, "{}/original.jpg".format(output_dir))
            save_image(scaled_image, "{}/bicubic.jpg".format(output_dir))
            save_image(
                scaled_y_image, "{}/bicubic_y.jpg".format(output_dir), is_rgb=False
            )
            save_image(
                output_y_image, "{}/result_y.jpg".format(output_dir), is_rgb=False
            )
            save_image(result_image, "{}/result.jpg".format(output_dir))

        return result_image

    def evaluate(self, filepath):
        input_image = align_image(load_image(filepath), self.scale)
        input_y_image = resize_image(convert_rgb_to_y(input_image), 1 / self.scale)
        input_scaled_y_image = resize_image(input_y_image, self.scale)

        output_y_image = self.run(input_y_image, input_scaled_y_image)
        ground_truth_y_image = convert_rgb_to_y(input_image)

        return calc_psnr_and_ssim(
            ground_truth_y_image, output_y_image, border=self.scale
        )

    def save(self, sess, name=""):
        filename = "{}.ckpt".format(name)
        saver = tf.train.Saver(max_to_keep=None)
        saver.save(sess, filename)

    def calc_metrics(self, files):
        psnrs = 0
        ssims = 0

        for file in files:
            psnr, ssim = self.evaluate(file)
            psnrs += psnr
            ssims += ssim

        psnr /= len(files)
        ssim = ssims / len(files)

        return psnr, ssim

    
    def metrics(self, output, labels):
        output_transposed = output if self.data_format == 'NHWC' else tf.transpose(output, perm=[0, 2, 3, 1])

        output = tf.Print(output, [tf.shape(output)], message="shape of output:", summarize=1000)
        output = tf.Print(output, [output], message="value of output:", summarize=1000)
        # labels = tf.Print(labels, [labels])
        # labels = tf.Print(labels)

        # labels = tf.image.rgb_to_yuv(labels)

        results = {}
        updates = []
        with tf.name_scope('metrics_cals'):
            mean_squared_error, mean_squared_error_update = tf.metrics.mean_squared_error(
                labels,
                output_transposed,
            )
            results["mean_squared_error"] = mean_squared_error
            updates.append(mean_squared_error_update)

            psnr_array = tf.image.psnr(labels, output, max_val=1.0)
            psnr, psnr_update = tf.metrics.mean(psnr_array)
            results["psnr"] = psnr
            updates.append(psnr_update)

            ssim_array = tf.image.ssim(labels, output, max_val=1.0)
            ssim, ssim_update = tf.metrics.mean(ssim_array)
            results["ssim"] = ssim
            updates.append(ssim_update)

            # merge all updates
            updates_op = tf.group(*updates)

            return results, updates_op
