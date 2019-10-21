"""
Script for transforming original datasets to OCMARTA be conform.

"""
from abc import ABC, abstractmethod
import argparse
from ast import literal_eval
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import yaml
import spectral as spy
import tifffile as tif
import numpy as np
import sys
import os
from os.path import join, dirname, abspath
import ast
from skimage import transform
from skimage.io import imread
import json


class Dataset(ABC):
    """Base class for datasets.
    """
    def __init__(self, satellite_id, in_path, generate_metadata, patch_size, stride, jobs, resolution, out_path, nodata_threshold, bands):
        self.in_path = in_path
        self.patch_size = patch_size
        self.stride = stride
        if self.stride is None:
            self.stride = self.patch_size
        self.jobs = jobs
        self.resolution = resolution
        self.out_path = out_path
        self.nodata_threshold = nodata_threshold
        self.metadata = self.get_satellite_data(satellite_id)
        
        # Convert bands into a list:
        self.bands = literal_eval(bands)


    @abstractmethod
    def process(self):
        """Start processing of dataset
        """
        pass
    
    def get_all_directories(self):
        for root, dirs, _ in os.walk(self.in_path):
            for directory in map(lambda d: join(root, d), dirs):
                yield directory
    
    @staticmethod
    def get_satellite_data(satellite):
        with open(join(abspath(dirname(__file__)), 'constants/sensors', satellite + '.yaml'), 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc
        
    def read(self, filename):
        return imread(filename)
        
    def normalise(self, band_with_id):
        idx, band = band_with_id
        return band
        
    @staticmethod
    def resize(image_and_scale):
        image, scale = image_and_scale
        if scale is None:
            return image

        image = transform.rescale(
            image,
            scale + 0.0001,
            anti_aliasing=True,
            mode='constant',
            preserve_range=True,
            multichannel=False
        )
        return image


    def load_bands(self, ids, files, resize=True):
        """Read, normalise, resize and combine bands
        
        Args:
            band_files: List of tuples of band identifier and file name. For example,
                `[('B01', '/path/B01.tif'), ('B02', '/path/B02.tif')]`.
                
        Returns:
            One numpy.array with all bands
        """
        metadata = self.metadata['bands']
        
        with ThreadPoolExecutor(self.jobs) as pool:
            # Load the data
            bands_data = pool.map(self.read, files)

            # Normalise the data
            bands_data = list(pool.map(self.normalise, zip(ids, bands_data)))

            # Resize the bands:
            if resize:
                scales = [
                    metadata[band_id]['resolution'] / self.resolution 
                    if metadata[band_id]['resolution'] != self.resolution else None
                    for band_id in ids
                ]
                bands_data = list(pool.map(
                    self.resize, zip(bands_data, scales)
                ))

        
        return np.stack(bands_data)


    def split_and_save(self, image, mask, metadata, tile_name, patch_size, stride):
        # TODO: Make this dataset independent
        n_x = image.shape[0]//patch_size
        n_y = image.shape[1]//patch_size
        step_x = stride
        step_y = stride

        for i in range(n_x):
            for j in range(n_y):
                mask_patch = mask[i*step_x:i*step_x+patch_size,
                                     j*step_y:j*step_y+patch_size, ...]

                # TODO: no data threshold could be dataset independent since "black" could be masked
                if np.mean(mask_patch[...,0]) <= nodata_threshold:
                    print('ROW:', str(i).zfill(2), '  COL:', str(j).zfill(2), end='\r')
                    image_patch = image[i*step_x:i*step_x+patch_size,
                             j*step_y:j*step_y+patch_size, ...]
                    patch_id = join(tile_name, str(i).zfill(3)+str(j).zfill(3))
                    self.save_patch(
                        patch_id, image_patch, mask_patch, metadata
                    )
                    
                            
    def save_patch(self, patch_id, image_patch, mask_patch, metadata):
        patch_path = join(self.out_path, patch_id)
        os.makedirs(patch_path, exist_ok=True)
        
        np.save(join(patch_path, 'image.npy'), image_patch.astype('float32'))
        np.save(join(patch_path, 'mask.npy'), mask_patch)
        
        if metadata is not None:
            with open(join(patch_path, 'metadata.json'),'w') as f:
                json.dump(metadata, f)

class L8Biome96(Dataset):
    def __init__(self, **kwargs):
        super().__init__(satellite_id='Landsat8', **kwargs)
    
    def process(self):
        tile_dirs = filter(self.is_valid_dir, self.get_all_directories())
        for tile_dir in tile_dirs:
            print(tile_dir)
        
    def is_valid_dir(self, directory):
        """
        Args:
            directory: Path to an hypothetical data directory.
        
        Returns:
            True if dir contains all band files and a .img envi mask file, and no subdirectories
        """
        children = os.listdir(directory)
        for i in range(1, 12):
            suffix = '_B'+str(i)+'.TIF'
            if not any(child.endswith(suffix) for child in children):
                return False
        if not any(child.endswith('fixedmask.img') for child in children):
            return False
        return True
        
    def read(self, filename):
        return tif.imread(filename)
    
class L8SPARCS(Dataset):
    def __init__(self, **kwargs):
        pass
    
class L7Irish206(Dataset):
    def __init__(self, **kwargs):
        super().__init__(satellite_id='Landsat7', **kwargs)
        print(self.metadata)
    
    def process(self):
        for directory in self.get_all_directories():
            # Check for band files:
            band_files = list(filter(
                lambda d: not d.endswith('_BQA.TIF'), 
                glob(join(directory, '*_B*.TIF'))
            ))
            if len(band_files) != 9:
                continue
            mask_file = glob(join(directory, '*mask*.TIF'))
            if not mask_file:
                continue
            mask_file = mask_file[0]
            
            print(directory)
                
            # We need the bands in a sorted ascending order:
            band_files = sorted(band_files)
            
            # Originally band names are 10, ..., 50, 61, 62, 70 for the Irish dataset. We 
            # standardise them here to make future processing easier.
            required_bands = {
                'B10': 'B1',
                'B20': 'B2',
                'B30': 'B3',
                'B40': 'B4',
                'B50': 'B5',
                'B61': 'B6_VCID_1',
                'B62': 'B6_VCID_2',
                'B70': 'B7',
                'B80': 'B8'
            }
            # We need a fixed order of the bands, to compose them later in a numpy array
            required_bands = OrderedDict(sorted(required_bands.items()))
            
            ids, band_files = zip(*[
                [required_bands[file.split('_')[-1][:-4]], file]
                for file in band_files
            ])
            ids = list(ids)
            
            # Sanity-check: are the bands from the directory, really the ones that we expect?
            if ids != list(required_bands.values()):
                print(directory, 'was skipped since not all required bands were found!')
                print('Missing bands:', set(required_bands.values()) - set(ids))
                continue
                
            print(self.load_bands(ids, band_files).shape)
            break
                
#             if any(os.path.isdir(child) for child in children):
#                 continue
#             for b in [10, 20, 30, 40, 50, 61, 62, 70, 80]:
#                 suffix = '_B{}.TIF'.format(b)
#                 if not any(child.endswith(suffix) for child in children):
#                     return False
#             return True
            
            
            
#             if generate_metadata:
#                 metadata = {'classes': ['FILL','SHADOW','CLEAR','THIN CLOUD','THICK CLOUD']}
#             else:
#                 metadata = None
#             print('Loading', os.path.split(tile_dir)[-1], '...')
#             img_arr, mask_arr = clean_tile(
#                 tile_dir, resolution, bands=bands, use_solar_elevation=use_solar_elevation,metadata=metadata)
#             print('\rLoaded', os.path.split(tile_dir)[-1])
#             tile_out_path = tile_dir.replace(biome_data_path,out_path)
#             os.makedirs(tile_out_path)
#             print('Splitting and saving', os.path.split(tile_dir)[-1], '...')
#             split_and_save(img_arr, mask_arr, tile_out_path,
#                            splitsize=splitsize, allow_overlap=allow_overlap,nodata_threshold=nodata_threshold,metadata=metadata)
#             print('\rSplit and saved', os.path.split(tile_dir)[-1])
#             band_files = glob(join(tile_dir, '*'))
    
class S2CESBIO(Dataset):
    def __init__(self, **kwargs):
        ...
    

if __name__ == '__main__':
    import shutil
    
    dataset_classes = {
        'L8Biome96': L8Biome96,
        'L8SPARCS': L8SPARCS,
        'L7Irish206': L7Irish206,
        'S2CESBIO': S2CESBIO,
    }

    arg_parser = argparse.ArgumentParser(
        description='Prepare different satellite for use as input into OCMARTA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument('in_path',
                            help='path to original dataset')
    arg_parser.add_argument('out_path', help='path to output directory')
    arg_parser.add_argument('-d', '--dataset', 
                            help='Name of dataset. Allowed are: ' + ', '.join(dataset_classes) + '.', 
                            default='L8Biome96', type=str)
    arg_parser.add_argument('-o', '--overwrite', 
                            help='Allows to overwrite an already existing output directory.', 
                            default=False, type=bool)
    arg_parser.add_argument('-r', '--resolution', type=float,
                            help='Resolution of outputs (metres)', default=30)
    arg_parser.add_argument('-p', '--patch_size', type=int,
                            help='Size of each output patch', default=256)
    arg_parser.add_argument('-t', '--nodata_threshold', type=float,
                            help='Fraction of patch that can have no-data values and still be used',
                            default=1.)
    arg_parser.add_argument('-b', '--bands', type=str,
                            help='List of bands to be used', default='None')
    arg_parser.add_argument('-s', '--stride', type=int, default=None,
                            help = 'Stride used for patch extraction. If this is smaller than patch_size, '
                                   'patches overlap each other. Default: set to patch_size, so no patches ' 
                                   'overlap each other.')
#     arg_parser.add_argument('-u', '--use_solar_elevation', type=bool,
#                             help='Whether to include sun-angle correction for reflectances',default=False)
    arg_parser.add_argument('-g', '--generate_metadata', type=bool,
                            help='Whether to generate a metadata file for each image/mask pair', default=False)
    arg_parser.add_argument('-j', '--jobs', type=int,
                            help='How many parallel jobs should be used to process the data.', default=4)


    kwargs = vars(arg_parser.parse_args())
    print('-' * 79)
    for key, value in kwargs.items():
        print('{:20}: {}'.format(key, value))
    print('-' * 79, '\n')
    
    overwrite = kwargs.pop('overwrite')
    if os.path.isdir(kwargs['out_path']):
        print('Output directory already exists.')
        if overwrite:
            print('Start overwriting...')
        else:
            print('Abort processing...')
            sys.exit()
    else:
        os.makedirs(kwargs['out_path'])
        
    # Create the specific dataset object:
    dataset_class = kwargs.pop('dataset')
    dataset = dataset_classes[dataset_class](
        **kwargs
    )
    dataset.process()
