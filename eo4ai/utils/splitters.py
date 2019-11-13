import numpy as np


class SlidingWindowSplitter:
    """Splitter for image/mask that takes evenly-spaced square patches.

    Attributes
    ----------
    patch_size : int
        Sidelength of extracted patches.
    stride : int
        Spacing between adjacent extractions.
    filters : list, optional
        All filters for patches that disallow certain criteria (default is None).
    """
    def __init__(self, patch_size, stride, filters = None):
        self.patch_size = patch_size
        self.stride = stride
        self.filters = filters

    def __call__(self,bands,mask):
        """Extracts evenly-spaced square patches from data.

        Parameters
        ----------
        bands : np.ndarray
            Band data for scene, aggregated into single 3D-array.
        mask : np.ndarray, bool
            One-hot encoded mask for scene.

        Returns
        -------
        bands_patches : list
            All extracted band data patches.
        mask_patches : list
            All extracted mask patches.
        patch_ids : list
            Identifiers for each patch based on x/y location.
        """
        n_x = (bands.shape[0]-self.patch_size) // self.stride + 1
        n_y = (bands.shape[1]-self.patch_size) // self.stride + 1
        step_x = self.stride
        step_y = self.stride
        #pre-allocated for efficiency
        bands_patches = np.empty((n_x*n_y,self.patch_size,self.patch_size,bands.shape[-1]))
        mask_patches = np.empty((n_x*n_y,self.patch_size,self.patch_size,mask.shape[-1]))
        patch_ids = []
        num_valid_patches=0
        for i in range(n_x):
            for j in range(n_y):
                region = slice(i*step_x,i*step_x+self.patch_size), \
                         slice(j*step_y,j*step_y+self.patch_size), ...

                bands_patch = bands[region]
                mask_patch = mask[region]

                if self.filters is not None:
                    if not all([filter(bands_patch,mask_patch) for filter in self.filters]):
                        continue

                bands_patches[num_valid_patches,...] = bands_patch
                mask_patches[num_valid_patches,...] = mask_patch
                patch_ids.append(str(i).zfill(3)+str(j).zfill(3))
                num_valid_patches+=1

        bands_patches = [bands_patches[i,...] for i in range(num_valid_patches)]
        mask_patches = [mask_patches[i,...].astype(bool) for i in range(num_valid_patches)]

        return bands_patches,mask_patches, patch_ids
