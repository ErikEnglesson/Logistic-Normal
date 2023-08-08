import logging
from typing import Callable, Dict, Optional, Sequence, Type, TypeVar, Union

from robustness_metrics.common import ops
from robustness_metrics.common import types
import tensorflow.compat.v2 as tf

import functools
from tensorflow.python.ops import stateless_random_ops
import numpy as np

import uncertainty_baselines as ub


# For datasets like UCI, the tf.data.Dataset returned by _read_examples will
# have elements that are Sequence[tf.Tensor], for TFDS datasets they will be
# Dict[Text, tf.Tensor] (types.Features), for Criteo they are a tf.Tensor.
PreProcessFn = Callable[
    [Union[int, tf.Tensor, Sequence[tf.Tensor], types.Features]],
    types.Features]

_BaseDatasetClass = Type[TypeVar('B', bound=ub.datasets.BaseDataset)]


def make_label_corrupted_dataset(dataset_cls: _BaseDatasetClass) -> _BaseDatasetClass:
    """Generate a BaseDataset with noisy labels."""

    class _LabelCorruptedBaseDataset(dataset_cls):
        """Add noisy labels."""

        def __init__(
                self,
                dataset: _BaseDatasetClass,
                severity: float = 0.4,
                corruption_type: str = 'asym',
                **kwargs):
            super().__init__(**kwargs)
            self.dataset = dataset
            self.severity = severity
            self.corruption_type = corruption_type
            assert corruption_type in [
                'aggre', 'worst', 'rand1', 'rand2', 'rand3', 'sym', 'asym', 'c100noise']
            self.is_synthetic = corruption_type in ['sym', 'asym']
            is_c10 = dataset.name == 'cifar10'
            self.num_classes = 10 if is_c10 else 100

            if not is_c10:
                assert corruption_type == 'c100noise' or corruption_type == 'sym' or corruption_type == 'asym'

            if not self.is_synthetic:
                paths_labels = './datasets/CIFAR-10_human_ordered.npy' if is_c10 else './datasets/CIFAR-100_human_ordered.npy'
                paths_indices = './datasets/image_order_c10_inverted.npy' if is_c10 else './datasets/image_order_c100_inverted.npy'
                noise_file = np.load(paths_labels, allow_pickle=True)

                # Clean labels are used for sanity checks.
                self.clean_labels = tf.convert_to_tensor(
                    noise_file.item().get('clean_label'))

                tf_to_torch = np.load(paths_indices)
                self.index_map = tf.convert_to_tensor(tf_to_torch)
                corruption_type_to_noise_key = {'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                                                'rand2': 'random_label2', 'rand3': 'random_label3', 'c100noise': 'noise_label'}
                noise_key = corruption_type_to_noise_key[corruption_type]

                self.noisy_labels = tf.convert_to_tensor(
                    noise_file.item().get(noise_key))
            else:
                self.clean_labels = None
                self.noisy_labels = None
                self.index_map = None

        def load(self,
                 *,
                 preprocess_fn=None,
                 batch_size: int = -1) -> tf.data.Dataset:
            if preprocess_fn:
                dataset_preprocess_fn = preprocess_fn
            else:
                dataset_preprocess_fn = (
                    self.dataset._create_process_example_fn())

            noisy_label_fn = _create_uniform_noisy_label_fn if self.corruption_type == 'sym' else _create_asym_noisy_label_fn
            noisy_label_fn = _create_real_noisy_label_fn if not self.is_synthetic else noisy_label_fn

            dataset_preprocess_fn = ops.compose(
                dataset_preprocess_fn,
                noisy_label_fn(self.num_classes, self._seed, self.severity, self.index_map, self.clean_labels, self.noisy_labels))
            dataset = self.dataset.load(
                preprocess_fn=dataset_preprocess_fn,
                batch_size=batch_size)

            return dataset

    return _LabelCorruptedBaseDataset


def _create_asym_noisy_label_fn(num_classes, seed, severity, index_map=None, clean_labels=None, noisy_labels=None) -> PreProcessFn:
    """Returns a function that adds an `noisy_labels` key to examles."""

    def _flip_label_c10(example: types.Features) -> types.Features:
        i = int(example['labels'])
        if i == 9:
            return 1.0
        # bird -> airplane
        elif i == 2:
            return 0.0
        # cat -> dog
        elif i == 3:
            return 5.0
        # dog -> cat
        elif i == 5:
            return 3.0
        # deer -> horse
        elif i == 4:
            return 7.0
        else:
            return float(i)

    def _flip_label_c100(example: types.Features) -> types.Features:
        i = int(example['labels'])
        return float((i+1) % 100)

    def _add_noisy_label(example: types.Features) -> types.Features:
        per_example_seed = tf.random.experimental.stateless_fold_in(
            seed, example['element_id'][0])
        random_func = functools.partial(
            stateless_random_ops.stateless_random_uniform, seed=per_example_seed)
        uniform_random = random_func(shape=[], minval=0, maxval=1.0)
        flip_cond = tf.math.less(uniform_random, severity)
        example['noisy_labels'] = tf.cond(flip_cond, lambda: _flip_label_c10(
            example) if num_classes == 10 else _flip_label_c100(example), lambda: example['labels'])
        return example

    return _add_noisy_label


def _create_uniform_noisy_label_fn(num_classes, seed, severity, index_map=None, clean_labels=None, noisy_labels=None) -> PreProcessFn:
    """Returns a function that adds an `noisy_labels` key to examles."""

    def _flip_label(example: types.Features, random_func) -> types.Features:
        return float(random_func(shape=[], minval=0, maxval=num_classes, dtype=tf.int32))

    def _add_noisy_label(example: types.Features) -> types.Features:
        per_example_seed = tf.random.experimental.stateless_fold_in(
            seed, example['element_id'][0])
        random_func = functools.partial(
            stateless_random_ops.stateless_random_uniform, seed=per_example_seed)
        uniform_random = random_func(shape=[], minval=0, maxval=1.0)
        flip_cond = tf.math.less(uniform_random, severity)
        example['noisy_labels'] = tf.cond(flip_cond, lambda: _flip_label(
            example, random_func), lambda: example['labels'])
        return example

    return _add_noisy_label


def _create_real_noisy_label_fn(num_classes, seed, severity, id_map, clean_labels, noisy_labels) -> PreProcessFn:
    """Returns a function that adds an `noisy_labels` key to examles."""

    def _add_noisy_label(example: types.Features) -> types.Features:
        element_id = example['id']
        tf.debugging.Assert(tf.strings.substr(
            element_id, 0, 6) == 'train_', [element_id])
        element_id = tf.strings.to_number(
            tf.strings.substr(element_id, 6, -1), out_type=tf.int64)
        mapped_id = id_map[element_id]
        clean_label = tf.cast(clean_labels[mapped_id], dtype=tf.float32)
        noisy_label = tf.cast(noisy_labels[mapped_id], dtype=tf.float32)
        tf.debugging.Assert(example['labels'] == clean_label, [
                            'not equal', element_id, mapped_id, example['labels'], clean_label])

        example['noisy_labels'] = noisy_label
        return example

    return _add_noisy_label
