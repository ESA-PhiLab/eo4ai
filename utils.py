import numpy as np
from abc import abstractmethod, ABC
import os
import math
import cv2
import skimage.io
import json
import pprint
import spectral as spy


#---ENCODERS---

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
        print(no_data.mean())
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


#---FILTERS---

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

#---FINDERS---

class FileFinderBySubStrings:
    """File finder for individual files based on substrings.

    Attributes
    ----------
    root_path : str
        Path to directory that will be searched within.
    """
    def __init__(self, root_path):
        self.root_path = root_path

    def __call__(self,substrings,startswith=None,endswith=None,dir_substrings=None):
        """Finds unique file based on substrings and other conditions.

        Parameters
        ----------
        substrings : list or str
            All strings that file must contain---not including directories or self.root_path.
        startswith : str, optional
            String that file must start with---not including directories or self.root_path (default is None).
        endswith : str, optional
            String that file must end with (default is None).
        dir_substrings : list or str, optional
            All strings that the directory-tree of file must contain (default is None).

        Returns
        -------
        str
            Path to unique file meeting conditions.
        """
        if isinstance(substrings,str):
            substrings = [substrings]
        if isinstance(dir_substrings,str):
            dir_substrings = [dir_substrings]

        found_paths = []
        for root,dirs,paths in os.walk(self.root_path):
            for possible_path in paths:
                if startswith is not None:
                    if not possible_path.startswith(startswith):
                        continue
                if endswith is not None:
                    if not possible_path.endswith(endswith):
                        continue
                if dir_substrings is not None:
                    if not all(substring in root for substring in dir_substrings):
                            continue
                if all(substring in possible_path for substring in substrings):
                    possible_path = os.path.join(root,possible_path)
                    found_paths.append(possible_path)
        if len(found_paths)==0:
            raise Exception('Failed to find any files matching criteria!')
        if len(found_paths)>1:
            raise Exception('Found multiple files matching criteria!')

        return found_paths[0]

class BandRegisterFinder:
    """File finder for multiple files based on substrings.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    root_path : str
        Path to directory that will be searched within.
    filefinder : FileFinderBySubStrings
        Finder for single files used to construct register.
    """
    def __init__(self, dataset_metadata,root_path):
        self.dataset_metadata = dataset_metadata
        self.root_path = root_path
        self.filefinder = FileFinderBySubStrings(self.root_path)

    def __call__(self,substrings='',startswith=None,endswith=None,dir_substrings=None):
        """Finds required files based on dataset_metadata, substrings, and other conditions.

        Parameters
        ----------
        substrings : list or str, optional
            All strings that all files must contain---not including directories or self.root_path (default is '').
        startswith : str, optional
            String that all files must start with---not including directories or self.root_path (default is None).
        endswith : str, optional
            String that all files must end with (default is None).
        dir_substrings : list or str, optional
            All strings that the directory-tree of all files must contain (default is None).

        Returns
        -------
        band_file_register : dict
            Dictionary mapping paths to the band data they contain as lists of band_ids.
        """
        if isinstance(substrings,str):
            substrings = [substrings]

        band_file_register={}
        for band_file_substrings,band_file_ids in self.dataset_metadata['band_files'].items():
            if isinstance(band_file_substrings,str):
                band_file_substrings = [band_file_substrings]

            path = self.filefinder(band_file_substrings+substrings,startswith=startswith,endswith=endswith,dir_substrings=dir_substrings)
            band_file_register[path] = [band_file_ids]

        return band_file_register


#---LOADERS---

class MultiFileBandLoader:
    """Loader for band data when contained in multiple files.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    imread : func or str
        Function or evaluatable string that reads some image file.
    """
    def __init__(self,dataset_metadata,imread=None):
        self.dataset_metadata = dataset_metadata
        if imread is None:
            self.imread = skimage.io.imread
        else:
            self.imread = imread

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

class SingleFileBandLoader:
    """Loader for band data when contained in single file.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    imread : func or str
        Function or evaluatable string that reads some image file.
    """
    def __init__(self,dataset_metadata,imread=None):
        self.dataset_metadata = dataset_metadata
        if imread is None:
            self.imread = skimage.io.imread
        else:
            self.imread = imread

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
            selected_idxs = [band_ids.index(band_id) for band_id in selected_bands]

            # Select bands and convert to list
            bands = [bands[...,idx] for idx in selected_idxs]
            return bands, selected_band_ids

class ImageLoader:
    """Loader for single images.

    Attributes
    ----------
    imread : func or str
        Function or evaluatable string that reads some image file.
    """
    def __init__(self,imread=None):
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
        if isinstance(self.imread,str):
            return eval(self.imread.format(path))
        return self.imread(path)

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


#---NORMALISERS---

class LandsatNormaliser(ABC):
    """Abstract Base Class for Landsat Normalisers.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    scene_metadata : dict or None
        Scene-specific Landsat metadata values.

    Methods
    -------
    _normalise_band(band,band_id,scene_metadata)
        Abstract method which should normalise a band when implemented.
    """
    def __init__(self,dataset_metadata):
        self.dataset_metadata = dataset_metadata
        self.scene_metadata = None

    def __call__(self,bands,band_ids,scene_metadata):
        """Normalises all bands based on metadata.

        Parameters
        ----------
        bands : list
            Band data from a Landsat scene.
        band_ids : list
            Identifiers for bands.
        scene_metadata : dict
            Scene-specific Landsat metadata values.

        Returns
        -------
        bands : list
            Normalised band data from a Landsat scene.
        """
        self.scene_metadata = scene_metadata
        if isinstance(band_ids,str): #Single-Band mode
            band_ids = [band_ids]
        for idx,band_id in enumerate(band_ids):
            bands[idx] = self._normalise_band(bands[idx],band_id,scene_metadata)
        return bands

    @abstractmethod
    def _normalise_band(self,band,band_id,scene_metadata):
        pass

class Landsat8Normaliser(LandsatNormaliser):
    """Normaliser for Landsat 8 data,to convert to TOA units."""
    def _normalise_band(self,band,band_id,scene_metadata):
        """Normalises any Landsat 8 band into TOA units

        Parameters
        ----------
        band : np.ndarray
            Array with Digital Number pixel values from Landsat 8.
        band_id : str
            Identifier for band being normalised.
        scene_metadata : dict
            Scene-specific Landsat metadata values.

        Returns
        -------
        band : np.ndarray
            Normalised band in TOA units.
        """
        bm = self.dataset_metadata['bands'][band_id] # Get a shortcut for the band's metadata

        gain = bm['gain']
        offset = bm['offset']

        band = band * gain + offset

        if bm['type'] == 'TOA Normalised Brightness Temperature':
            band = (bm['K2']  / np.log(bm['K2'] / band + 1))
            band = (band - bm['MINIMUM_BT']) / (bm['MAXIMUM_BT'] - bm['MINIMUM_BT'])

        if bm.get('solar_correction', False):
            band /= math.sin(float(self.scene_metadata['SUN_ELEVATION'])*math.pi/180)

        return band

class Landsat7Pre2011Normaliser(LandsatNormaliser):
    """Normaliser for Landsat7 Pre-2011 data format, to convert to TOA units."""
    def _normalise_band(self,band,band_id,scene_metadata):
        """Normalises any Landsat 7 pre-2011 band into TOA units

        Parameters
        ----------
        band : np.ndarray
            Array with Digital Number pixel values from Landsat 7.
        band_id : str
            Identifier for band being normalised.
        scene_metadata : dict
            Scene-specific Landsat metadata values.

        Returns
        -------
        band : np.ndarray
            Normalised band in TOA units.
        """
        bm = self.dataset_metadata['bands'][band_id] # Get a shortcut for the band's metadata

        QCAL_MAX = bm['QCAL_MAX']
        if isinstance(QCAL_MAX,str):
            QCAL_MAX = self.scene_metadata[QCAL_MAX]
        QCAL_MIN = bm['QCAL_MIN']
        if isinstance(QCAL_MIN,str):
            QCAL_MIN = self.scene_metadata[QCAL_MIN]
        L_MAX = bm['L_MAX']
        if isinstance(L_MAX,str):
            L_MAX = self.scene_metadata[L_MAX]
        L_MIN = bm['L_MIN']
        if isinstance(L_MIN,str):
            L_MIN = self.scene_metadata[L_MIN]

        radiance = ((L_MAX-L_MIN)/(QCAL_MAX-QCAL_MIN))*(band-QCAL_MIN) + L_MIN

        if bm['type'] == 'TOA Reflectance':
            ESUN = bm['ESUN']
            band = math.pi*radiance/ESUN

            if bm.get('solar_correction', False):
                band /= math.sin(float(self.scene_metadata['SUN_ELEVATION'])*math.pi/180)

        if bm['type'] == 'TOA Normalised Brightness Temperature':
            band = (bm['K2']  / np.log(bm['K2'] / radiance + 1))
            band = (band - bm['MINIMUM_BT']) / (bm['MAXIMUM_BT'] - bm['MINIMUM_BT'])
        return band

#---ORGANISERS---

class BySceneAndPatchOrganiser:
    """Organiser for output files based on scenes and patch ids."""
    def __init__(self):
        pass
    def __call__(self,out_path,scene_id,patch_ids):
        """Organises output directory structure for all samples taken from a scene.

        Parameters
        ----------
        out_path : str
            Parent directory for outputted dataset.
        scene_id : str
            Identifier for scene.
        patch_ids : list
            Identifiers for all samples taken from scene.

        Returns
        -------
        list
            Output directories corresponding to each entry in patch_ids.
        """
        return [os.path.join(out_path,scene_id,patch_id) for patch_id in patch_ids]

#---RESIZERS---

class BandsMaskResizer:
    """Resizer for bands and mask, by changing their resolution.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    to_array : bool, optional
        Whether to convert resized bands into single array (default is True).
    strict : bool, optional
        Whether to disallow resizing operations that lead to bands of different sizes (default is True).
    """
    def __init__(self,dataset_metadata,to_array=True,strict=True):
        self.dataset_metadata = dataset_metadata
        self.to_array = to_array
        self.strict = strict

    def __call__(self,bands,band_ids,mask,resolution):
        """Resizes bands and mask to single resolution.

        Parameters
        ----------
        bands : list
            List of band arrays for a scene.
        band_ids : list
            Identifiers corresponding to bands.
        mask : np.ndarray, bool
            One-hot array.
        resolution : int
            Target resolution to resize to.

        Returns
        -------
        rescaled_bands : np.ndarray or list
            Resized band data at new resolution.
        rescaled_mask : np.ndarray, bool
            Rescaled one-hot array at new resolution.
        """
        # BANDS
        band_target_sizes,mask_target_size = self._get_target_sizes(bands,band_ids,mask,resolution)
        rescaled_bands = []
        for band,band_id,target_size in zip(bands,band_ids,band_target_sizes):
            if not all(np.array(band.shape[:2])==np.array(target_size)):
                rescaled_bands.append(cv2.resize(band,target_size[::-1],cv2.INTER_CUBIC))
            else:
                rescaled_bands.append(band)

        # MASK
        if not all(np.array(mask.shape[:2])==np.array(mask_target_size)):
            rescaled_mask = cv2.resize(mask.astype(np.uint8),mask_target_size[::-1],cv2.INTER_NEAREST).astype(bool)
        else:
            rescaled_mask = mask

        if self.to_array:
            rescaled_bands = np.moveaxis(np.array(rescaled_bands),0,-1)
        return rescaled_bands,rescaled_mask

    def _get_target_sizes(self,bands,band_ids,mask,resolution):
        band_scale_factors = [self.dataset_metadata['bands'][b]['resolution']/resolution for b in band_ids]
        band_target_sizes = [(round(b.shape[0]*s),round(b.shape[1]*s)) for b,s in zip(bands,band_scale_factors)]
        mask_scale_factor = self.dataset_metadata['mask']['resolution']/resolution
        mask_target_size = (round(mask.shape[0]*mask_scale_factor),round(mask.shape[1]*mask_scale_factor))
        if self.strict and len(set(band_target_sizes))>1:
            raise Exception('Resizing operation leads to inconsistent band shapes, which is disallowed when "strict" is True.')

        elif self.to_array and len(set(band_target_sizes))>1:
            most_common = max(set(band_target_sizes), key = band_target_sizes.count)
            band_target_sizes = [most_common]*len(band_target_sizes)
        return band_target_sizes,mask_target_size

#---SAVERS---

class Saver(ABC):
    def __init__(self,overwrite=False):
        self.overwrite=overwrite

    def __call__(self,*args):

        self._make_dirs(args[-1])
        for items in zip(*args):
            self._save_sample(*items)

    @abstractmethod
    def _save_sample(self,*args,**kwargs):
        pass

    def _make_dirs(self,out_paths):
        for path in out_paths:
            if not os.path.isdir(path):
                os.makedirs(path)

class ImageMaskNumpySaver(Saver):
    def _save_sample(self,image,mask,out_path):
        image_path = os.path.join(out_path,'image.npy')
        mask_path = os.path.join(out_path,'mask.npy')
        if not self.overwrite:

            if not os.exists(image_path):
                np.save(image_path,image)
            if not os.exists(mask_path):
                np.save(mask_path,mask)
        else:
            np.save(image_path,image)
            np.save(mask_path,mask)



class ImageMaskDescriptorNumpySaver(ImageMaskNumpySaver):
    def __call__(self,images,masks,descriptors,out_paths):
        self._make_dirs(out_paths)
        if not isinstance(descriptors,list):
            descriptors = [descriptors]*len(out_paths)
        for item in zip(images,masks,descriptors,out_paths):
            self._save_sample(*item)
    def _save_sample(self,image,mask,descriptors,out_path):
        print(os.path.split(out_path)[1])
        super()._save_sample(image,mask,out_path)
        descriptors_path = os.path.join(out_path,'descriptors.npy')
        if not self.overwrite:
            if not os.exists(descriptors_path):
                np.save(descriptors_path,descriptors)
        else:
            np.save(descriptors_path,descriptors)

class MetadataJsonSaver(Saver):
    def __call__(self,metadata,out_paths):
        for path in out_paths:
            self._save_sample(metadata,path)

    def _save_sample(self,metadata,out_path):
        metadata_path = os.path.join(out_path,'metadata.json')
        if not self.overwrite:
            if not os.exists(metadata_path):
                with open(metadata_path,'w') as f:
                    json.dump(metadata,f)
        else:
            with open(metadata_path,'w') as f:
                json.dump(metadata,f)

#---SPLITTERS---

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
        # BUG: n_x,n_y are defined by //patch_size but should be function of stride.
        n_x = bands.shape[0] // self.patch_size
        n_y = bands.shape[1] // self.patch_size
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
        mask_patches = [mask_patches[i,...] for i in range(num_valid_patches)]

        return bands_patches,mask_patches, patch_ids

#---WRITERS---

class LandsatMetadataWriter:
    """Writer for output metadata writing of Landsat datasets.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    sun_elevation : bool, optional
        Whether to include scene-specific value for sun_elevation (defaut is False).
    """
    def __init__(self,dataset_metadata,sun_elevation=False):
        self.dataset_metadata = dataset_metadata
        self.sun_elevation = sun_elevation

    def __call__(self,scene_metadata,scene_id,band_ids,class_ids,**kwargs):
        """Create dictionary for all relevant metadata values for a Landsat scene.

        Parameters
        ----------
        scene_metadata : dict
            Scene-specific Landsat metadata values.
        scene_id : str
            Identifier for scene.
        band_ids : list
            Identifiers for bands.
        class_ids : list
            Identifiers for classes.
        **kwargs
            All other (name, value) pairs to be saved

        Returns
        -------
        output_metadata : dict
            Scene-specific output metadata.
        """
        output_metadata = {
                'spacecraft_id': self.dataset_metadata['spacecraft_id'],
                'scene_id': scene_id,
                'bands':band_ids,
                'band_types':[
                    self.dataset_metadata['bands'][band]['type']
                    for band in band_ids
                ],
                'band_centres': [
                    self.dataset_metadata['bands'][band]['band_centre']
                    for band in band_ids
                ],
                'band_widths': [
                    self.dataset_metadata['bands'][band]['band_width']
                    for band in band_ids
                ],
                'classes':class_ids
            }
        if self.sun_elevation:
            output_metadata['sun_elevation'] = scene_metadata['SUN_ELEVATION']

        for band in band_ids:
            descriptive_names = [
                                 self.dataset_metadata['bands'][band_id].get('descriptive_name')
                                 for band_id in band_ids
                                ]
            band_positions = [list(self.dataset_metadata['bands'].keys()).index(band_id) for band_id in band_ids]
            named_bands = {
                           descriptive_name: band_pos
                           for descriptive_name, band_pos in zip(descriptive_names,band_positions)
                           if descriptive_name is not None
                          }


            output_metadata['named_bands'] = named_bands

        #Add any extra user-supplied metadata
        output_metadata.update(kwargs)
        return output_metadata
