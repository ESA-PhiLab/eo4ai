import copy
import numpy as np

from eo4ai.utils import filters


def test_FilterByBandValue():

    bands1 = np.zeros([56, 47, 10])
    bands2 = np.ones([56, 47, 4])
    bands3 = copy.copy(bands1)
    bands3[..., 0] = 1
    bands4 = copy.copy(bands2)
    bands4[..., 0] = 0
    bands5 = copy.copy(bands2)
    bands5[0, 0, 0] = 0
    bands6 = copy.copy(bands2)
    bands6[0, 0, :] = 0
    bands7 = copy.copy(bands1)
    bands7[5, 5, 2] = 1
    bands8 = copy.copy(bands1)
    bands8[5, 5, :] = 1
    bands9 = np.concatenate((bands1, bands2), axis=2)
    bands10 = np.concatenate((
                        np.ones((20, 20, 4)),
                        np.zeros((20, 10, 4))
                        ), axis=1)

    mask = None

    filter1 = filters.FilterByBandValue()
    filter2 = filters.FilterByBandValue(target_value=1)
    filter3 = filters.FilterByBandValue(mode='any')
    filter4 = filters.FilterByBandValue(mode='all', threshold=0.3)
    filter5 = filters.FilterByBandValue(mode='all', threshold=0.8)

    assert not filter1(bands1, mask)
    assert filter1(bands2, mask)
    assert filter1(bands3, mask)
    assert filter1(bands4, mask)
    assert filter1(bands5, mask)
    assert not filter1(bands6, mask)
    assert not filter1(bands7, mask)
    assert not filter1(bands8, mask)

    assert filter2(bands1, mask)
    assert not filter2(bands2, mask)
    assert filter2(bands3, mask)
    assert filter2(bands4, mask)
    assert not filter2(bands5, mask)
    assert not filter2(bands6, mask)
    assert filter2(bands7, mask)
    assert not filter2(bands8, mask)

    assert not filter3(bands1, mask)
    assert filter3(bands2, mask)
    assert not filter3(bands3, mask)
    assert not filter3(bands4, mask)
    assert not filter3(bands5, mask)
    assert not filter3(bands6, mask)
    assert not filter3(bands7, mask)
    assert not filter3(bands8, mask)

    assert filter4(bands9, mask)
    assert not filter4(bands10, mask)

    assert filter5(bands9, mask)
    assert filter5(bands10, mask)


def test_FilterByMaskClass():

    bands = None
    mask0 = np.concatenate(
                        (np.ones((10, 10, 1)), np.zeros((10, 10, 1))),
                        axis=2
                        )
    mask1 = np.concatenate(
                        (np.zeros((10, 10, 1)), np.ones((10, 10, 1))),
                        axis=2
                        )

    mask2 = copy.copy(mask0)
    mask2[2, 4, :] = [0, 1]
    mask3 = copy.copy(mask1)
    mask3[5, 3, :] = [1, 0]
    mask4 = np.concatenate((mask0, mask1), axis=1)
    mask5 = np.concatenate((mask0, mask0, mask1), axis=1)
    mask6 = np.concatenate((mask1, mask0, mask1), axis=1)
    mask7 = np.zeros((10, 10, 4))

    filter0 = filters.FilterByMaskClass(target_index=0)
    filter1 = filters.FilterByMaskClass(target_index=1)
    filter2 = filters.FilterByMaskClass(target_index=0, threshold=0.5)
    filter3 = filters.FilterByMaskClass()
    assert not filter0(bands, mask0)
    assert filter0(bands, mask1)
    assert not filter0(bands, mask2)
    assert not filter0(bands, mask3)
    assert not filter0(bands, mask4)

    assert filter1(bands, mask0)
    assert not filter1(bands, mask1)
    assert not filter1(bands, mask2)
    assert not filter1(bands, mask3)
    assert not filter1(bands, mask4)

    assert not filter2(bands, mask0)
    assert filter2(bands, mask1)
    assert not filter2(bands, mask2)
    assert filter2(bands, mask3)
    assert filter2(bands, mask4)
    assert not filter2(bands, mask5)
    assert filter2(bands, mask6)

    assert filter3(bands, mask0)
    assert filter3(bands, mask1)
    assert not filter3(bands, mask7)
