import numpy as np


class FilterByBandValue:
    """Patch filter that disallows based on presence of a given value in the bands.

    Attributes
    ----------
    target_value
        Value in bands by which filtering occurs.
    mode : str
        Controller for which multiband_func to use, can be 'all' or 'any'.
    threshold : float
        Maximum allowable amount of target class between 0 and 1.
    multiband_func : func
        Function that decides whether pixel is counted based on multiple bands.
    """
    def __init__(self, target_value=None, mode='all',threshold=None):
        self.target_value = target_value
        if self.target_value is None:
            self.target_value = 0

        self.mode = mode
        if self.mode not in ['all','any']:
            raise ValueError('Invalid mode - "{}"'.format(self.mode))

        if self.mode=='all':
            self.multiband_func = np.all
        elif self.mode=='any':
            self.multiband_func = np.any

        self.threshold = threshold
        if self.threshold is None:
            self.threshold = 0

    def __call__(self, bands, mask):
        """Calculate whether patch passes filter based on class in mask.

        Parameters
        ----------
        bands : np.ndarray
            Array of band values.
        mask : np.ndarray
            One-hot encoded array corresponding to classes.

        Returns
        -------
        bool
            True if patch has passed filter, False otherwise.
        """
        is_target = self.multiband_func(bands==self.target_value, axis=-1)
        if np.mean(is_target)>self.threshold:
            return False #Patch has failed

        return True #Patch has passed

class FilterByMaskClass:
    """Patch filter that disallows based on presence of a given class in the mask.

    Attributes
    ----------
    target_index
        Index in mask by which filtering occurs (last dimension of mask).
    threshold : float
        Maximum allowable amount of target class between 0 and 1.
    """
    def __init__(self, target_index = None, threshold = None):
        self.target_index = target_index
        self.threshold = threshold
        if self.threshold is None:
            self.threshold = 0
    def __call__(self, bands, mask):
        """Calculate whether patch passes filter based on class in mask.

        Parameters
        ----------
        bands : np.ndarray
            Array of band values.
        mask : np.ndarray
            One-hot encoded array corresponding to classes.

        Returns
        -------
        bool
            True if patch has passed filter, False otherwise.
        """
        if self.target_index is not None:  #Use specific class as no_data mask
            is_target = mask[...,self.target_index]
        else:   #Use places for which there are no labels as no_data
            is_target = np.all(mask==0, axis=-1)

        if np.mean(is_target)>self.threshold:
            return False #Patch has failed

        return True #Patch has passed

class FilterBySaturation:
    """Patch filter that disallows based on presence of saturation in bands.

    Attributes
    ----------
    saturation_value :
        Index in mask by which filtering occurs (last dimension of mask).
    threshold : float
        Maximum allowable amount of target class between 0 and 1.
    """
    def __init__(self, saturation_value = None, threshold = None):
        self.saturation_value = saturation_value
        self.threshold = threshold
        if self.threshold is None:
            self.threshold = 0

    def __call__(self, bands, mask):
        """Calculate whether patch passes filter based on saturation in bands.

        Parameters
        ----------
        bands : np.ndarray
            Array of band values.
        mask : np.ndarray, bool
            One-hot encoded array corresponding to classes.

        Returns
        -------
        bool
            True if patch has passed filter, False otherwise.
        """
        if self.saturation_value is not None:
            saturated = bands >= self.saturation_value
        else:
            saturation_value = np.nanmax(bands) - np.finfo(np.float32).eps
            saturated = bands >= saturation_value

        if np.mean(saturated)>self.threshold:
            return False #Patch has failed

        return True #Patch has passed
