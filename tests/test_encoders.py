import numpy as np
import pytest

from eo4ai.utils import encoders


class StubEncoder(encoders.Encoder):
    def __call__(self):
        pass


def test_Encoder():

    good_metadata = {
        'mask': {
                'classes': {
                      'CLASS1': [1],
                      'CLASS2': [0],
                      'CLASS3': [-3],
                      'CLASS4': [1.1]
                      },
                'another_subfield': 9
                },
        'another_field': True,
        }
    bad_metadata1 = {'not_mask': {'classes': {
      'CLASS1': [1],
      'CLASS2': [0],
      'CLASS3': [-3],
      'CLASS4': [1.1]
      }}}

    bad_metadata2 = {'mask': {'classes': [{
      'CLASS1': [1],
      'CLASS2': [0],
      'CLASS3': [-3],
      'CLASS4': [1.1]
      }]
      }}

    bad_metadata3 = {'mask': {'classes': 'not_a_dict'}}

    bad_metadata4 = None
    bad_metadata5 = [1, 3, 6]
    bad_metadata6 = 'string value'

    StubEncoder(good_metadata)
    with pytest.raises(KeyError):
        StubEncoder(bad_metadata1)
    with pytest.raises(AttributeError):
        StubEncoder(bad_metadata2)
        StubEncoder(bad_metadata3)
    with pytest.raises(TypeError):
        StubEncoder(bad_metadata4)
        StubEncoder(bad_metadata5)
        StubEncoder(bad_metadata6)


def test_MapByColourEncoder():
    dataset_metadata1 = {'mask': {'classes': {
      'CLASS1': [0, 0, 1],
      'CLASS2': [0, 0, 128],
      'fsdgnw': [0, 0.2, 255],
      'CLASS4': [0, 255, 255],
      'CLASS5': [128, 128, 0],
      'aldefs': [128, 128, 128],
      'CLASS7': [127, 128, 128]
      }}}

    dataset_metadata2 = {'mask': {'classes': {
      'CLASS1': [1],
      'CLASS2': [0],
      'CLASS3': [-3],
      'CLASS4': [1.1]
      }}}

    encoder1 = encoders.MapByColourEncoder(dataset_metadata1)
    encoder2 = encoders.MapByColourEncoder(dataset_metadata2)

    mask1_shape = [2, 8]
    mask2_shape = [101, 99]
    mask1 = np.reshape([
            list(dataset_metadata1['mask']['classes'].values())[idx]
            for idx in np.random.choice(
                np.arange(len(dataset_metadata1['mask']['classes'].values())),
                np.prod(mask1_shape)
                )
            ],
            [*mask1_shape, -1]
            )
    mask2 = np.reshape([
            list(dataset_metadata2['mask']['classes'].values())[idx]
            for idx in np.random.choice(
                np.arange(len(dataset_metadata2['mask']['classes'].values())),
                np.prod(mask2_shape)
                )
            ],
            [*mask2_shape, -1]
            )

    enc_mask1, class_ids1 = encoder1(mask1)
    enc_mask2, class_ids2 = encoder2(mask2)
    # Every pixel has exactly one class
    assert np.unique(np.sum(enc_mask1, axis=-1)) == 1
    assert np.unique(np.sum(enc_mask2, axis=-1)) == 1

    # Shape is preserved
    assert enc_mask1.shape[:2] == mask1.shape[:2]
    assert enc_mask2.shape[:2] == mask2.shape[:2]

    # Number of classes is preserved
    assert enc_mask1.shape[-1] == len(
                                dataset_metadata1['mask']['classes'].values()
                                )
    assert enc_mask2.shape[-1] == len(
                                dataset_metadata2['mask']['classes'].values()
                                )

    # Type of array is bool
    assert enc_mask1.dtype == np.bool
    assert enc_mask2.dtype == np.bool

    # Classes are preserved
    assert class_ids1 == tuple(dataset_metadata1['mask']['classes'].keys())
    assert class_ids2 == tuple(dataset_metadata2['mask']['classes'].keys())


def test_MapByValueEncoder():
    dataset_metadata = {'mask': {'classes': {
      'CLASS1': 3,
      'CLASS2': 0.1,
      'CLASS3': -23,
      'CLASS4': 9,
      'CLASS5': 0,
      'CLASS6': 2,
      'CLASS7': 1
      }}}

    encoder = encoders.MapByValueEncoder(dataset_metadata)

    mask_shape = [79, 14]
    mask = np.random.choice(
                        list(dataset_metadata['mask']['classes'].values()),
                        size=mask_shape
                        )

    enc_mask, class_ids = encoder(mask)

    # Every pixel has exactly one class
    assert np.unique(np.sum(enc_mask, axis=-1)) == 1

    # Shape is preserved
    assert enc_mask.shape[:2] == mask.shape[:2]

    # Number of classes is preserved
    assert enc_mask.shape[-1] == len(
                                dataset_metadata['mask']['classes'].values()
                                )

    # Type of array is bool
    assert enc_mask.dtype == np.bool

    # Classes are preserved
    assert class_ids == tuple(dataset_metadata['mask']['classes'].keys())
