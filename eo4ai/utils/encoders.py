from abc import abstractmethod, ABC
import os
import numpy as np


class Encoder(ABC):
    """Abstract Base Class for Encoders.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    classes : dict
        Class/value pairs for dataset annotations.
    class_ids : list
        Class names for dataset annotations.
    patterns : list
        Numerical values in the mask that correspond to the class_ids.

    Methods
    -------
    __call__(*args,**kwargs)
        Abstract method that should encode the mask somehow when implemented.
    """
    def __init__(self,dataset_metadata):
        self.dataset_metadata = dataset_metadata
        self.classes = self.dataset_metadata['mask']['classes']
        self.class_ids,self.patterns = zip(*self.classes.items())

    @abstractmethod
    def __call__(self,*args,**kwargs):
        pass

class MapByColourEncoder(Encoder):
    """Encoder used to map 3D arrays into one-hot encoded arrays by matching colours."""
    def __call__(self,mask):
        """Encodes the mask by the colour of a given pixel into one-hot array.

        Parameters
        ----------
        mask : np.ndarray
            2D array of original labelled values for a scene.

        Returns
        -------
        encoded_mask : np.ndarray, bool
            One-hot encoded array corresponding to classes.
        class_ids : list
            Class names corresponding to position on final axis of encoded_mask.
        """
        #Match each colour in self.classes with pixels in mask
        encoded_mask = np.stack([np.all(mask[:,:]==colour,axis=-1) for colour in self.patterns],axis=-1).astype(np.bool)
        return encoded_mask,self.class_ids


class MapByValueEncoder(Encoder):
    """Encoder used to map 2D arrays into one-hot encoded arrays by matching values."""
    def __call__(self,mask):
        """Encodes the mask by the value of a given pixel into one-hot array.

        Parameters
        ----------
        mask : np.ndarray
            2D array of original labelled values for a scene.

        Returns
        -------
        encoded_mask : np.ndarray, bool
            One-hot encoded array corresponding to classes.
        class_ids : list
            Class names corresponding to position on final axis of encoded_mask.
        """
        encoded_mask = np.stack([mask[:,:]==value for value in self.patterns],axis=-1).astype(np.bool)
        return encoded_mask,self.class_ids

class L7IrishEncoder(Encoder):
    """
    Will be removed if/when USGS fixes issues with Irish dataset masks.

    Methods
    -------
    __call__(mask)
        Encodes the mask based on specific requirements of Landsat 7 Irish dataset.

    """
    def __init__(self,dataset_metadata):
        self.dataset_metadata = dataset_metadata
        self.class_ids = ['FILL','SHADOW','CLEAR','THIN','THICK']

    def __call__(self,mask,bands):
        """Encodes the mask based on specific requirements of Landsat 7 Irish dataset.

        Parameters
        ----------
        mask : np.ndarray
            2D array of original labelled values for a scene.
        bands : list
            List of band arrays for a scene.

        Returns
        -------
        encoded_mask : np.ndarray, bool
            One-hot encoded array corresponding to classes.
        class_ids : list
            Class names corresponding to position on final axis of encoded_mask.
        """
        no_data = np.zeros(mask.shape[:2],dtype=bool)
        for b in bands:
            if np.all(b.shape == mask.shape):
                no_data += b==0
        encoded_mask = np.empty((*mask.shape[:2],5))
        if mask[0,0] == 0:
            # Class and pixel value:
            encoded_mask[..., 0] = no_data  # FILL (no data)
            encoded_mask[..., 1] = False      # SHADOW (there is no shadow class defined)
            encoded_mask[..., 2] = mask == 128  # CLEAR
            encoded_mask[..., 3] = mask == 192  # THIN
            encoded_mask[..., 4] = mask == 255  # THICK
        else:
            # Pixel values for FILL and CLEAR are ambiguous. So we have to check whether
            # there is actual data in other bands:
            encoded_mask[..., 0] = no_data    # FILL (no data)
            encoded_mask[..., 1] = mask == 0  # SHADOW
            encoded_mask[..., 2] = (mask == 255) & ~no_data  # CLEAR
            encoded_mask[..., 3] = mask == 192  # THIN
            encoded_mask[..., 4] = mask == 128  # THICK


        return encoded_mask.astype(np.bool),self.class_ids
