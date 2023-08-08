# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
   LN using a LeNet-5 on (Fashion) MNIST.

   This code is based on the implementation in baselines/mnist/deterministic.py 
   in the Uncertainty Baselines repository:
   https://github.com/google/uncertainty-baselines
"""


import os
from absl import app
from absl import flags
from absl import logging
import edward2 as ed
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import uncertainty_baselines as ub
import utils  # local file import
import robustness_metrics as rm
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


flags.DEFINE_enum('dataset', 'mnist',
                  enum_values=['mnist', 'fashion_mnist'],
                  help='Name of the image dataset.')
flags.DEFINE_enum('noise_type', 'sym',
                  enum_values=['sym', 'asym'],
                  help='Name of the image dataset.')
flags.DEFINE_float('noise_rate', 0.4, 'Percent of flipped labels.')
flags.DEFINE_integer('ensemble_size', 1, 'Number of ensemble members.')
flags.DEFINE_boolean('bootstrap', False,
                     'Sample the training set for bootstrapping.')
flags.DEFINE_integer('training_steps', 17579, 'Training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('validation_freq', 5, 'Validation frequency in steps.')
flags.DEFINE_string('output_dir', '/tmp/det_training',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')


flags.DEFINE_float('label_smoothing', 0.01,
                   'Label smoothing parameter in (0,1].')
flags.register_validator('label_smoothing',
                         lambda ls: ls > 0.0 and ls <= 1.0,
                         message='--label_smoothing must be in (0, 1].')
flags.DEFINE_bool('single_scale', False,
                  'Whether to output full diagonal or single value.')
flags.DEFINE_float('temperature', 1.0,
                   'Temperature for heteroscedastic head.')
flags.DEFINE_float('min_scale', 0.5,
                   'Minimum value for diagonal of Sigma_L.')

flags.DEFINE_integer('num_mc_samples', 2000,
                     'Num MC samples for heteroscedastic layer.')


FLAGS = flags.FLAGS


def get_smoothed_targets(y_true, num_classes):
    one_hot_labels = tf.one_hot(
        tf.cast(tf.squeeze(y_true), tf.int32), num_classes+1)
    smoothed_targets = (1.0-FLAGS.label_smoothing) * one_hot_labels + \
        FLAGS.label_smoothing * \
        tf.ones(tf.shape(one_hot_labels)) / (num_classes+1)
    return smoothed_targets


def symmetric_noise(labels, percent):
    noisy_labels = labels.copy()
    indices = np.random.permutation(len(labels))
    for i, idx in enumerate(indices):
        if i < percent * len(labels):
            noisy_labels[idx] = np.random.randint(10, dtype=np.int32)

    return noisy_labels


def asymmetric_noise(labels, percent):
    noisy_labels = labels.copy()
    for i in range(10):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            if j < percent * len(indices):
                if i == 7:
                    noisy_labels[idx] = 1
                elif i == 2:
                    noisy_labels[idx] = 7
                elif i == 5:
                    noisy_labels[idx] = 6
                elif i == 6:
                    noisy_labels[idx] = 5
                elif i == 3:
                    noisy_labels[idx] = 8

    return noisy_labels


def lenet5(input_shape, num_classes):
    """Builds LeNet5."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(6,
                                   kernel_size=5,
                                   padding='SAME',
                                   activation='relu')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding='SAME')(conv1)
    conv2 = tf.keras.layers.Conv2D(16,
                                   kernel_size=5,
                                   padding='SAME',
                                   activation='relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding='SAME')(conv2)
    conv3 = tf.keras.layers.Conv2D(120,
                                   kernel_size=5,
                                   padding='SAME',
                                   activation=tf.nn.relu)(pool2)
    flatten = tf.keras.layers.Flatten()(conv3)
    dense1 = tf.keras.layers.Dense(84, activation=tf.nn.relu)(flatten)

    loc_sigma = tf.keras.layers.Dense(
        num_classes*2, name='loc_sigma', activation=None)(dense1)
    output = tfp.layers.DistributionLambda(lambda t: create_logitnormal(t[..., 0:num_classes],  # mu
                                                                        t[..., num_classes:num_classes*2], FLAGS.min_scale, num_classes)
                                           )(loc_sigma)

    return tf.keras.Model(inputs=inputs, outputs=output)


def create_logitnormal(loc, scale, min_scale, num_classes):
    loc = tf.ensure_shape(loc, [None, num_classes])
    scale = tf.ensure_shape(scale, [None, num_classes])
    scale = tf.reshape(scale, [-1, num_classes, 1])
    diag = min_scale * tf.ones([tf.shape(loc)[0], num_classes])

    mvn = tfd.MultivariateNormalDiagPlusLowRank(loc=loc,
                                                scale_diag=diag,
                                                scale_perturb_factor=scale,
                                                validate_args=False)  # Debug

    bijector = tfb.Chain(
        [tfb.SoftmaxCentered(), tfb.Scale(1.0 / FLAGS.temperature)])
    logit_normal = tfd.TransformedDistribution(
        distribution=mvn,
        bijector=bijector,
        name='LogitNormalTransformedDistribution')

    return logit_normal


def main(argv):
    del argv  # unused arg
    if not FLAGS.use_gpu:
        raise ValueError('Only GPU is currently supported.')
    if FLAGS.num_cores > 1:
        raise ValueError('Only a single accelerator is currently supported.')
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)
    tf.io.gfile.makedirs(FLAGS.output_dir)

    if FLAGS.dataset == 'mnist':
        dataset_builder_class = ub.datasets.MnistDataset
    else:
        dataset_builder_class = ub.datasets.FashionMnistDataset
    n_total = 50000
    n_train = 45000
    train_dataset = next(dataset_builder_class(
        'train').load(batch_size=n_total).as_numpy_iterator())
    x_train = train_dataset['features']
    y_train = train_dataset['labels']

    # Create train-validation splits
    np.random.seed(FLAGS.seed)
    inds = np.random.permutation(n_total)
    inds_train, inds_val = inds[0:n_train], inds[n_train:]
    x_val, y_val = x_train[inds_val], y_train[inds_val]
    x_train, y_train = x_train[inds_train], y_train[inds_train]

    # Noisy Labels
    if FLAGS.noise_rate > 0.0:
        if FLAGS.noise_type == 'sym':
            y_train_noisy = symmetric_noise(y_train, FLAGS.noise_rate)
            y_val_noisy = symmetric_noise(y_val, FLAGS.noise_rate)
        elif FLAGS.noise_type == 'asym':
            y_train_noisy = asymmetric_noise(y_train, FLAGS.noise_rate)
            y_val_noisy = asymmetric_noise(y_val, FLAGS.noise_rate)
    else:
        y_train_noisy = y_train.copy()
        y_val_noisy = y_val.copy()

    print(sum(y_train == y_train_noisy) / float(n_train))
    print(y_train_noisy[0:5])
    print(y_train[0:5])

    test_dataset = next(dataset_builder_class(
        'test').load(batch_size=10000).as_numpy_iterator())
    x_test = test_dataset['features']
    y_test = test_dataset['labels']
    num_classes = int(np.amax(y_train)) + 1

    # Note that we need to disable v2 behavior after we load the data.
    tf1.disable_v2_behavior()

    assert FLAGS.ensemble_size == 1
    ensemble_filenames = []
    for i in range(FLAGS.ensemble_size):
        # TODO(trandustin): We re-build the graph for each ensemble member. This
        # is due to an unknown bug where the variables are otherwise not
        # re-initialized to be random. While this is inefficient in graph mode, I'm
        # keeping this for now as we'd like to move to eager mode anyways.
        model = lenet5(x_train.shape[1:], num_classes)

        def negative_log_likelihood(y, rv_y):
            del rv_y  # unused arg

            ln = model.output
            one_hot_labels = tf.one_hot(
                tf.cast(tf.squeeze(y), tf.int32), num_classes+1)
            smoothed_targets = (1.0-FLAGS.label_smoothing) * one_hot_labels + \
                FLAGS.label_smoothing * \
                tf.ones(tf.shape(one_hot_labels)) / (num_classes+1)

            return -ln.log_prob(tf.squeeze(smoothed_targets))  # pylint: disable=cell-var-from-loop

        def accuracy(y_true, y_sample):
            del y_sample  # unused arg
            loc = model.output.distribution.loc
            probs = model.output.bijector.forward(loc, axis=1)

            return tf.equal(
                tf.argmax(input=probs, axis=1),
                tf.cast(tf.squeeze(y_true), tf.int64))

        def log_likelihood(y_true, y_sample):
            del y_sample  # unused arg
            ln = model.output
            one_hot_labels = tf.one_hot(
                tf.cast(tf.squeeze(y_true), tf.int32), num_classes+1)
            smoothed_targets = (1.0-FLAGS.label_smoothing) * one_hot_labels + \
                FLAGS.label_smoothing * \
                tf.ones(tf.shape(one_hot_labels)) / (num_classes+1)

            return ln.log_prob(tf.squeeze(smoothed_targets))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate),
            loss=negative_log_likelihood,
            metrics=[log_likelihood, accuracy])
        member_dir = os.path.join(FLAGS.output_dir, '')
        tensorboard = tf1.keras.callbacks.TensorBoard(
            log_dir=member_dir,
            update_freq=FLAGS.batch_size * FLAGS.validation_freq)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=member_dir + 'best-model.weights',
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        if FLAGS.bootstrap:
            inds = np.random.choice(n_train, n_train, replace=True)
            x_sampled = x_train[inds]
            y_sampled = y_train_noisy[inds]

        epochs = (FLAGS.batch_size * FLAGS.training_steps) // n_train
        model.fit(
            x=x_train if not FLAGS.bootstrap else x_sampled,
            y=y_train_noisy if not FLAGS.bootstrap else y_sampled,
            batch_size=FLAGS.batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val_noisy),
            validation_freq=max(
                (FLAGS.validation_freq * FLAGS.batch_size) // n_train, 1),
            verbose=1,
            callbacks=[tensorboard, model_checkpoint_callback])

        member_filename = os.path.join(member_dir, 'model.weights')
        ensemble_filenames.append(member_filename)
        model.save_weights(member_filename)

    labels = tf.keras.layers.Input(shape=y_train.shape[1:])
    ll = tf.keras.backend.function([model.input, labels], [
        model.output.bijector.forward(model.output.distribution.loc, axis=1),
        model.output.distribution.loc,
        model.output.distribution.covariance(),
        model.output.bijector.inverse(
            get_smoothed_targets(labels, 10), axis=1),
    ])

    last_weights = [member_filename]
    best_weights = [os.path.join(member_dir, 'best-model.weights')]

    last_metrics_vals = {
        'train': utils.ln_metrics(
            x_train, y_train, model, ll, weight_files=last_weights),
        'train_noisy': utils.ln_metrics(
            x_train, y_train_noisy, model, ll, weight_files=last_weights, y_true=y_train, log_dir=member_dir),
        'val': utils.ln_metrics(
            x_val, y_val, model, ll, weight_files=last_weights),
        'val_noisy': utils.ln_metrics(
            x_val, y_val_noisy, model, ll, weight_files=last_weights),
        'test': utils.ln_metrics(
            x_test, y_test, model, ll, weight_files=last_weights),
    }

    best_metrics_vals = {
        'train': utils.ln_metrics(
            x_train, y_train, model, ll, weight_files=best_weights),
        'train_noisy': utils.ln_metrics(
            x_train, y_train_noisy, model, ll, weight_files=best_weights),
        'val': utils.ln_metrics(
            x_val, y_val, model, ll, weight_files=best_weights),
        'val_noisy': utils.ln_metrics(
            x_val, y_val_noisy, model, ll, weight_files=best_weights),
        'test': utils.ln_metrics(
            x_test, y_test, model, ll, weight_files=best_weights),
    }

    tensorboard.writer.reopen()
    for split, metrics in last_metrics_vals.items():
        logging.info(split)
        for metric_name in metrics:
            logging.info('%s: %s', metric_name, metrics[metric_name])
        logs = {'last-' + split + '/' + k: v for k, v in metrics.items()}
        tensorboard._write_custom_summaries(epochs, logs=logs)

    for split, metrics in best_metrics_vals.items():
        logging.info(split)
        for metric_name in metrics:
            logging.info('%s: %s', metric_name, metrics[metric_name])
        logs = {'best-' + split + '/' + k: v for k, v in metrics.items()}
        tensorboard._write_custom_summaries(epochs, logs=logs)

    tensorboard.writer.close()


if __name__ == '__main__':
    app.run(main)
