import numpy as np
import os
import pytest

from eo4ai.datasets import Dataset


class DummyDataset(Dataset):
    """Dummy dataset class used in testing to get metadata from all datasets.
    Useful for checking dataset metadata."""
    def get_scenes(self):
        pass

    def process_scene(self):
        pass


@pytest.fixture
def dummy_dataset():
    def _make_dataset(dataset_id):
        return DummyDataset(dataset_id, 1)
    return _make_dataset


@pytest.fixture
def all_dummy_datasets():
    all_dataset_ids = os.listdir(os.path.join(
                                        os.path.dirname(__file__),
                                        'constants',
                                        'datasets'
                                        ))
    all_datasets = {id: DummyDataset(id, 1) for id in all_dataset_ids}
    return all_datasets


class DataGenerator(DummyDataset):
    def __init__(self, dataset_id, selected_band_ids=None,
                 DN_range=None):
        super().__init__(
                    dataset_id=dataset_id,
                    jobs=1,
                    selected_band_ids=selected_band_ids
                    )
        self.DN_range = DN_range
        if self.selected_band_ids is None:
            self.selected_band_ids = list(
                                        self.dataset_metadata['bands'].keys()
                                        )
        if self.DN_range is None:
            self.DN_range = (1, 256)

    def __call__(self, size):
        bands, band_ids = self._generate_bands(size)
        mask = self._generate_mask(size)
        return bands, band_ids, mask

    def _generate_bands(self, size):
        band_res = [
                self.dataset_metadata['bands'][id]['resolution']
                for id in self.selected_band_ids
                ]
        bands = [self._generate_band(size, r, self.DN_range) for r in band_res]
        return bands, self.selected_band_ids

    def _generate_band(self, size, res, DN_range):
        X = int(size[0]/res)
        Y = int(size[1]/res)
        min_DN, max_DN = DN_range
        band = np.random.randint(low=min_DN, high=max_DN, size=(X, Y))
        return band

    def _generate_mask(self, size):
        res = self.dataset_metadata['mask']['resolution']
        X = int(size[0]/res)
        Y = int(size[1]/res)
        classes = list(self.dataset_metadata['mask']['classes'].values())
        mask = np.array(classes)[np.random.randint(0, len(classes), X*Y)]
        mask = np.reshape(mask, (X, Y, -1))
        return mask


if __name__ == '__main__':
    gen = DataGenerator('L8SPARCS80')
    BS, ids, mask = gen((6000, 6000))
    for b, id in zip(BS, ids):
        print(id, b.shape, b.min(), b.max())
    print(mask.shape)
