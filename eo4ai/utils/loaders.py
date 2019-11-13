import numpy as np
import skimage.io


class ImageLoader:
    """Loader for single images.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    imread : func
        Function that reads some image file.
    """
    def __init__(self,dataset_metadata,imread=None):
        self.dataset_metadata = dataset_metadata
        if imread is None:
            self.imread = skimage.io.imread
        else:
            self.imread = imread

    def __call__(self,path):
        """Loads image from file.

        Parameters
        ----------
        path : str
            Path to image file.

        Returns
        -------
        np.ndarray
            Image data loaded from path.
        """
        return self.imread(path)

class MultiFileBandLoader(ImageLoader):
    """Loader for band data when contained in multiple files."""

    def __call__(self,band_file_register,selected_band_ids=None):
        """Loads all required band files

        Parameters
        ----------
        band_file_register : dict
            Dictionary mapping paths to the band data they contain as lists of band_ids.
        selected_band_ids : list, optional
            List of band_ids to load (default is None).

        Returns
        -------
        bands : list
            All loaded bands.
        band_ids : list
            Band ids corresponding order to loaded bands.
        """
        band_ids = []
        bands = []
        if selected_band_ids is not None:
            band_file_register = dict([item for item in band_file_register.items() if any([band in item[1] for band in selected_band_ids])])
        for band_file,file_band_ids in band_file_register.items():
            file_data = self.imread(band_file)

            if file_data.ndim==2:
                file_bands = [file_data]
            else:
                file_bands = [file_data[...,i] for i in range(file_data.shape[-1])]
            band_ids += file_band_ids
            bands += file_bands

        if selected_band_ids is None:
            return bands, band_ids
        else:
            selected_idxs = [band_ids.index(band_id) for band_id in selected_band_ids]
            bands = [band for i,band in enumerate(bands) if i in selected_idxs]
            return bands, selected_band_ids

class SingleFileBandLoader(ImageLoader):
    """Loader for band data when contained in single file."""

    def __call__(self,path,file_id,selected_band_ids=None):
        """Loads band data from file and then selects bands.

        Parameters
        ----------
        path : str
            Path to band data file
        file_id : str
            Identifier corresponding to file in question in dataset_metadata
        selected_band_ids : list, optional
            List of band_ids to keep once loaded (default is None).

        Returns
        -------
        bands : list
            All loaded bands.
        band_ids : list
            Band ids corresponding order to loaded bands.

        """
        band_ids = self.dataset_metadata['band_files'][file_id]
        bands = self.imread(path)
        if selected_band_ids is None:
            # convert to list
            bands = [bands[...,i] for i in range(bands.shape[-1])]
            return bands, band_ids
        else:
            selected_idxs = [band_ids.index(band_id) for band_id in selected_band_ids]

            # Select bands and convert to list
            bands = [bands[...,idx] for idx in selected_idxs]
            return bands, selected_band_ids



class SimpleSpectralDescriptorsLoader:
    """Loader for simple spectral descriptors.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    """
    def __init__(self,dataset_metadata):
        self.dataset_metadata = dataset_metadata

    def __call__(self,band_ids=None):
        """Calculates simple spectral descriptors for the given bands.

        Parameters
        ----------
        band_ids : list, optional
            Identifiers for bands for which descriptors should be calculated

        Returns
        -------
        descriptors : np.ndarray
            Triplets of values for each band (min. wavelen, centre wavelen, max. wavelen).
        """
        if band_ids is None:
            band_ids = list(self.dataset_metadata['bands'].keys())

        descriptors = []
        for band in band_ids:
            band_centre = self.dataset_metadata['bands'][band]['band_centre']
            band_width = self.dataset_metadata['bands'][band]['band_width']
            descriptors.append([band_centre-band_width/2,band_centre,band_centre+band_width/2])

        descriptors = np.array(descriptors)
        return descriptors

class LandsatMTLLoader:
    """Loader for Landsat metadata files, into non-hierarchical dictionary."""
    def __init__(self):
        pass

    def __call__(self,path):
        """Loads metadata values from a given path to Landsat metadata file.

        Parameters
        ----------
        path : str
            Path to Landsat metadata file.

        Returns
        -------
        config : dict
            Dictionary containing all values from Landsat metadata file (non-hierarchical).
        """
        with open(path) as f:
            config =  {
                entry[0]: entry[1]
                for entry in map(lambda l: "".join(l.split()).split('='), f)
                if len(entry) == 2
            }
        for k,v in config.items():
            try:
                config[k] = float(v)
            except:
                continue
        return config


class Sentinel2MTLLoader:
    """Loader for Landsat metadata files, into non-hierarchical dictionary."""
    def __init__(self):
        pass

    def __call__(self,path):
        """Loads metadata values from a given path to Landsat metadata file.

        Parameters
        ----------
        path : str
            Path to Landsat metadata file.

        Returns
        -------
        config : dict
            Dictionary containing all values from Landsat metadata file (non-hierarchical).
        """
        with open(path) as f:
            config =  {
                entry[0]: entry[1]
                for entry in map(lambda l: "".join(l.split()).split('='), f)
                if len(entry) == 2
            }
        for k,v in config.items():
            try:
                config[k] = float(v)
            except:
                continue
        return config
