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

class MapByColourEncoder:
    def __init__(self,dataset_metadata):
        self.dataset_metadata = dataset_metadata
        self.classes = self.dataset_metadata['mask']['classes']
        self.class_ids,self.colours = zip(*self.classes.items())

    def __call__(self,mask):
        #Match each colour in self.classes with pixels in mask
        encoded_mask = np.stack([np.all(mask[:,:]==colour,axis=-1) for colour in self.colours],axis=-1)
        return encoded_mask,self.class_ids


class MapByValueEncoder:
    def __init__(self,dataset_metadata):
        self.dataset_metadata = dataset_metadata
        self.classes = self.dataset_metadata['mask']['classes']
        self.class_ids,self.values = zip(*self.classes.items())


    def __call__(self,mask):
        encoded_mask = np.stack([mask[:,:]==value for value in self.values],axis=-1)
        return encoded_mask,self.class_ids

class L7IrishEncoder:
    """
    Will be removed if/when USGS fixes issues with Irish dataset masks
    """
    def __init__(self,dataset_metadata):
        self.dataset_metadata = dataset_metadata
        self.class_ids = ['FILL','SHADOW','CLEAR','THIN','THICK']

    def __call__(self,mask,bands):
        no_data = np.zeros(mask.shape[:2],dtype=bool)
        for b in bands:
            if np.all(b.shape == mask.shape):
                no_data += b==0
        print(no_data.mean())
        new_mask = np.empty((*mask.shape[:2],5))
        if mask[0,0] == 0:
            # Class and pixel value:
            new_mask[..., 0] = no_data  # FILL (no data)
            new_mask[..., 1] = False      # SHADOW (there is no shadow class defined)
            new_mask[..., 2] = mask == 128  # CLEAR
            new_mask[..., 3] = mask == 192  # THIN
            new_mask[..., 4] = mask == 255  # THICK
        else:
            # Pixel values for FILL and CLEAR are ambiguous. So we have to check whether
            # there is actual data in other bands:
            new_mask[..., 0] = no_data    # FILL (no data)
            new_mask[..., 1] = mask == 0  # SHADOW
            new_mask[..., 2] = (mask == 255) & ~no_data  # CLEAR
            new_mask[..., 3] = mask == 192  # THIN
            new_mask[..., 4] = mask == 128  # THICK


        return new_mask,self.class_ids


#---FILTERS---

class NoDataFilterByBands:

    def __init__(self, no_data_value=None, mode='all',threshold=None):
        self.no_data_value = no_data_value
        if self.no_data_value is None:
            self.no_data_value = 0

        self.mode = mode
        if self.mode not in ['all','any']:
            raise ValueError('Invalid mode - "{}"'.format(self.mode))

        if self.mode=='all':
            self.no_data_func = np.all
        elif self.mode=='any':
            self.no_data_func = np.any

        self.threshold = threshold
        if self.threshold is None:
            self.threshold = 0


    def __call__(self, bands, mask):

        no_data = self.no_data_func(bands==self.no_data_value, axis=-1)
        if np.mean(no_data)>self.threshold:
            return False #Patch has failed

        return True #Patch has passed

class NoDataFilterByMask:

    def __init__(self, no_data_index = None, threshold = None):
        self.no_data_index = no_data_index
        self.threshold = threshold
        if self.threshold is None:
            self.threshold = 0
    def __call__(self, bands, mask):

        if self.no_data_index is not None:  #Use specific class as no_data mask
            no_data = mask[...,self.no_data_index]
        else:   #Use places for which there are no labels as no_data
            no_data = np.all(mask==0, axis=-1)

        if np.mean(no_data)>self.threshold:
            return False #Patch has failed

        return True #Patch has passed

class SaturationFilter:

    def __init__(self, saturation_value = None, threshold = None):
        self.saturation_value = saturation_value
        self.threshold = threshold
        if self.threshold is None:
            self.threshold = 0

    def __call__(self, bands, mask):

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

    def __init__(self, root_path):
        self.root_path = root_path

    def __call__(self,substrings,startswith=None,endswith=None,in_dir=None):
        if isinstance(substrings,str):
            substrings = [substrings]
        found_paths = []
        for root,dirs,paths in os.walk(self.root_path):
            for possible_path in paths:
                if startswith is not None:
                    if not possible_path.startswith(startswith):
                        continue
                if endswith is not None:
                    if not possible_path.endswith(endswith):
                        continue
                if in_dir is not None:
                    if in_dir not in root:
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
    def __init__(self, dataset_metadata,root_path):
        self.dataset_metadata = dataset_metadata
        self.root_path = root_path
        self.filefinder = FileFinderBySubStrings(self.root_path)

    def __call__(self,substrings=None,startswith=None,endswith=None,in_dir=None):
        band_file_register={}
        for band_file_substring,band_file_ids in self.dataset_metadata['band_files'].items():
            path = self.filefinder(band_file_substring,startswith=startswith,endswith=endswith,in_dir=in_dir)
            band_file_register[path] = [band_file_ids]
        pprint.pprint(band_file_register)
        return band_file_register


#---LOADERS---

class MultiImageBandLoader:
    def __init__(self,dataset_metadata,imread=None):
        self.dataset_metadata = dataset_metadata
        if imread is None:
            self.imread = skimage.io.imread
        else:
            self.imread = imread

    def __call__(self,band_file_register,selected_band_ids=None):

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

class SingleImageBandLoader:
    def __init__(self,dataset_metadata,imread=None):
        self.dataset_metadata = dataset_metadata
        if imread is None:
            self.imread = skimage.io.imread
        else:
            self.imread = imread

    def __call__(self,path,file_id,selected_band_ids=None):
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

    def __init__(self,imread=None):
        if imread is None:
            self.imread = skimage.io.imread
        else:
            self.imread = imread

    def __call__(self,path):
        if isinstance(self.imread,str):
            return eval(self.imread.format(path))
        return self.imread(path)

class SimpleSpectralDescriptorsLoader:

    def __init__(self,dataset_metadata):
        self.dataset_metadata = dataset_metadata

    def __call__(self,bands=None):

        if bands is None:
            bands = list(self.dataset_metadata['bands'].keys())

        descriptors = []
        for band in bands:
            band_centre = self.dataset_metadata['bands'][band]['band_centre']
            band_width = self.dataset_metadata['bands'][band]['band_width']
            descriptors.append([band_centre-band_width/2,band_centre,band_centre+band_width/2])

        descriptors = np.array(descriptors)
        return descriptors

class LandsatMTLLoader:
    def __init__(self):
        pass

    def __call__(self,path):
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
    def __init__(self,dataset_metadata):
        self.dataset_metadata = dataset_metadata
        self.scene_metadata = None

    def __call__(self,bands,band_ids,scene_metadata):
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

    def _normalise_band(self,band,band_id,scene_metadata):

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

    def _normalise_band(self,band,band_id,scene_metadata):

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
    def __init__(self):
        pass
    def __call__(self,out_path,scene_id,patch_ids):
        return [os.path.join(out_path,scene_id,patch_id) for patch_id in patch_ids]

#---RESIZERS---

class BandsMaskResizer:
    def __init__(self,dataset_metadata,to_array=False,strict=True):
        self.dataset_metadata = dataset_metadata
        self.to_array = to_array
        self.strict = strict

    def __call__(self,bands,band_ids,mask,resolution):
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
        print(rescaled_mask.shape,rescaled_bands.shape)
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
    def __init__(self, patch_size, stride, filters = None):
        self.patch_size = patch_size
        self.stride = stride
        self.filters = filters

    def __call__(self,bands,mask):
        print(bands.shape)
        print(mask.shape)
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

        print(bands_patches.shape)

        bands_patches = [bands_patches[i,...] for i in range(num_valid_patches)]
        mask_patches = [mask_patches[i,...] for i in range(num_valid_patches)]

        print(len(bands_patches))

        return bands_patches,mask_patches, patch_ids

#---WRITERS---

class LandsatMetadataWriter:
    def __init__(self,dataset_metadata,sun_elevation=False):
        self.dataset_metadata = dataset_metadata
        self.sun_elevation = sun_elevation

    def __call__(self,scene_metadata,scene_id,band_ids,class_ids,**kwargs):
        self.output_metadata = {
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
            self.output_metadata['sun_elevation'] = scene_metadata['SUN_ELEVATION']

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


            self.output_metadata['named_bands'] = named_bands

        #Add any extra user-supplied metadata
        self.output_metadata.update(kwargs)
        return self.output_metadata
