"""
Script for transforming original datasets to OCMARTA be conform.

"""
from abc import ABC, abstractmethod
import argparse
from ast import literal_eval
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from itertools import repeat
import math

import cv2 as cv
import yaml
import spectral as spy
import tifffile as tif
import numpy as np
import sys
import os
from os.path import join, dirname, abspath, split
import ast
from skimage import transform
from skimage.io import imread
import json

# TODO:
#   1. Add classes.yaml for each satellite, find way to do this for Irish which
#        has inconsistent classes. Read self.classes and metadata['classes']
#        from here, not hardcoded.
#
#   2. Fix descriptors, currently +/- full bandwidth instead of half
#
#   3. Separate reading from encoding of mask, put encoding in abstract base class.
#
#   4. Move dataset_bands to a dataset/bands.yaml. No hardcoding of band names here.
#
#   5. Order methods more systematically

class ReadingError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super(ReadingError, self).__init__(message)


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
        self.generate_metadata = generate_metadata
        self.satellite_id = satellite_id

        # We have two metadata variables: satellite and tile metadata.
        # TM should change for each new tile.
        self.sm = self.get_sm(satellite_id)
        self.tm = None

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
    def get_sm(satellite):
        with open(join(abspath(dirname(__file__)), 'constants/sensors', satellite + '.yaml'), 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc

    def get_descriptors(self, bands):
        descriptors = np.empty((len(bands), 3), dtype=np.float32)
        for i, band in enumerate(bands):
            centre = self.sm['bands'][band]['band_centre']
            width = self.sm['bands'][band]['band_width']
            descriptors[i, :] = [centre-width, centre, centre+width]

        return descriptors

    def get_metadata(self, bands):
        return {
            'band_centres': [
                self.sm['bands'][band]['band_centre']
                for band in bands
            ],
            'band_widths': [
                self.sm['bands'][band]['band_width']
                for band in bands
            ],
            'resolution': self.resolution,
            'satellite_id': self.satellite_id,
            'bands': bands,
            'classes': list(self.classes.keys()),
            'named_band': None,
            'sun_elevation': None,
            'band_types': [
                self.sm['bands'][band]['type']
                for band in bands
            ],
        }

    def read(self, filename):
        image = imread(filename)
        if not image.shape[0]:
            raise ReadingError(f'Cannot load {filename}')
        return image

    def normalise(self, band_with_id):
        idx, band = band_with_id
        return band

    @staticmethod
    def resize(image_and_target_size):
        image, target_size = image_and_target_size
        if image.shape == target_size:
            return image

        # opencv works with width x height but numpy arrays with
        # rows (height) x columns (width):
        image = cv.resize(
            image, target_size[::-1],
            cv.INTER_CUBIC
        )
        return image

    @staticmethod
    def resize_mask(mask, target_size):
        if mask.shape == target_size:
            return mask

        # opencv works with width x height but numpy arrays with
        # rows (height) x columns (width) therefore we reverse target_size.
        # We need to keep the unique values of mask so we disable any interpolation
        # and use the nearest neighbour instead.
        print(target_size)
        mask = cv.resize(
            mask, target_size[::-1],cv.INTER_NEAREST)
        return mask


    def load_bands(self, ids, files):
        """Read, resize, normalise and combine bands

        Args:
            ids: List of band identifiers.
            files: List of band file names. The order will be kept when the bands are
                stacked onto each other. For example, `['/path/B01.tif', '/path/B02.tif')]`.


        Returns:
            A numpy.array with all bands and a no data mask (True where no data).
        """

        with ThreadPoolExecutor(self.jobs) as pool:
            # Load the data
            bands = list(pool.map(self.read, files))

            # Resize the bands. To make sure all bands end up with the same size, we calculate
            # the target size for the first bands only and use it for all other bands.
            if self.sm['bands'][ids[0]]['resolution'] != self.resolution:
                scale_factor = self.sm['bands'][ids[0]]['resolution'] / self.resolution
                target_size = bands[0].shape[0] * scale_factor, bands[0].shape[1] * scale_factor
            else:
                target_size = bands[0].shape

            bands = list(pool.map(
                self.resize, zip(bands, repeat(target_size))
            ))

            # Create a mask for no data
            no_data = ~np.all(bands, axis=0)

            # Normalise the data
            bands = list(pool.map(self.normalise, zip(ids, bands)))

            # We need the channels at the last dimension
            bands = np.moveaxis(np.stack(bands), 0, -1)

            # make sure that no data stays no data:
            bands[no_data, :] = 0

        return bands, no_data

    def split_and_save(self, tile_name, bands, mask, no_data, metadata=None, descriptors=None):
        n_x = bands.shape[0] // self.patch_size
        n_y = bands.shape[1] // self.patch_size
        step_x = self.stride
        step_y = self.stride

        valid_patches = 0
        skipped_patches = 0

        for i in range(n_x):
            for j in range(n_y):
                region = slice(i*step_x,i*step_x+self.patch_size), \
                         slice(j*step_y,j*step_y+self.patch_size), ...
                mask_patch = mask[region]

                if np.mean(no_data[region[:2]]) <= self.nodata_threshold:

                    image_patch = bands[region]
                    bmax = np.nanmax(image_patch[..., 0]) - np.finfo(np.float32).eps
                    if np.nanmean(image_patch[...,0] >= bmax) > 0.25:
                        skipped_patches += 1
                        continue

                    print('ROW:', str(i).zfill(2), '  COL:', str(j).zfill(2), end='\r')

                    patch_id = join(tile_name, str(i).zfill(3)+str(j).zfill(3))
                    self.save_patch(
                        patch_id, image_patch, mask_patch, metadata, descriptors
                    )
                    valid_patches += 1

        print(f'Summary: {n_x*n_y} potential, {valid_patches} valid and {skipped_patches} oversaturated patches')

    def save_patch(self, patch_id, image_patch, mask_patch, metadata=None, descriptors=None):
        patch_path = join(self.out_path, patch_id)
        os.makedirs(patch_path, exist_ok=True)

        np.save(join(patch_path, 'image.npy'), image_patch.astype('float32'))
        np.save(join(patch_path, 'mask.npy'), mask_patch.astype(bool))

        if descriptors is not None:
            np.save(join(patch_path, 'descriptors.npy'), descriptors.astype('float32'))

        if self.generate_metadata and metadata is not None:
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



class L8SPARCS80(Dataset):
    def __init__(self, **kwargs):
        self.dataset_bands = OrderedDict([
            ('B1', 'B1'),
            ('B2', 'B2'),
            ('B3', 'B3'),
            ('B4', 'B4'),
            ('B5', 'B5'),
            ('B6', 'B6'),
            ('B7', 'B7'),
            # SPARCS does not contain 'panchromatic' B8
            ('B9', 'B9'),
            ('B10', 'B10'),
            ('B11', 'B11'),
        ])
        super().__init__(satellite_id='Landsat8',**kwargs)
        self.L8SPARCS80_resolution = 30
        self.mask_vals = []
        self.classes = self.get_classes()

    @staticmethod
    def get_classes():
        with open(join(abspath(dirname(__file__)),'constants', 'datasets','Landsat8_SPARCS80','classes.yaml'), 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc

    def process(self):
        # The user might want to have only some bands:
        if self.bands is not None:
            required_bands = OrderedDict([
                (old, new)
                for old, new in self.dataset_bands.items()
                if new in self.bands
            ])
        else:
            required_bands = self.dataset_bands

        descriptors = self.get_descriptors(required_bands.values())
        metadata = self.get_metadata(list(required_bands.values()))
        metadata['named_band'] = {
            'RED': 'B4',
            'GREEN': 'B3',
            'BLUE': 'B2',
        }

        tile_names = self.get_tile_names()
        for tile_name in tile_names:
            descriptors = self.get_descriptors(required_bands.values())
            metadata = self.get_metadata(list(required_bands.values()))
            metadata['named_band'] = {
                'RED': required_bands['B4'],
                'GREEN': required_bands['B3'],
                'BLUE': required_bands['B2']
            }
            # Get current tile metadata:
            self.tm = self.get_tm(tile_name)
            if self.tm is None:
                print('Tile skipped since no tile metadata was found!')
                continue

            metadata['sun_elevation'] = self.tm['SUN_ELEVATION']
            bands = self.load_bands(tile_name,required_bands)
            mask  = self.load_mask(tile_name)
            no_data = np.zeros(mask.shape[:2])
            self.split_and_save(tile_name, bands, mask, no_data, metadata, descriptors)


    def get_tile_names(self):
        return list(set([file[:21] for file in os.listdir(self.in_path)]))


    def get_tm(self, tile_name):
        filename = join(self.in_path, tile_name+'_mtl.txt')
        if not filename:
            return None
        with open(filename) as file:
            return {
                entry[0]: entry[1]
                for entry in map(lambda l: "".join(l.split()).split('='), file)
                if len(entry) == 2
            }

    def load_bands(self,tile_name,required_bands):
        band_file = glob(join(self.in_path,tile_name+'*_data.tif'))
        # TODO Raise error if there is not exactly one file in band_file
        band_file = band_file[0]
        bands = tif.imread(band_file)
        band_idxs = [i for i,k in enumerate(self.dataset_bands.keys()) if k in required_bands.keys()]
        bands = bands[...,band_idxs]

        if self.L8SPARCS80_resolution != self.resolution:
            scale_factor = self.L8SPARCS80_resolution / self.resolution
            target_size = int(bands.shape[0] * scale_factor), int(bands.shape[1] * scale_factor)
            bands = self.resize(bands,target_size)

        bands = self.normalise(bands,required_bands)
        return bands

    def load_mask(self, tile_name):
        band_file = glob(join(self.in_path,tile_name+'*_mask.png'))[0]
        mask = imread(band_file)
        if self.L8SPARCS80_resolution != self.resolution:
            scale_factor = self.L8SPARCS80_resolution / self.resolution
            target_size = int(mask.shape[0] * scale_factor), int(mask.shape[1] * scale_factor)
            mask = self.resize_mask(mask,target_size)

        new_mask = np.stack([np.all(mask[:,:]==colour,axis=-1) for colour in self.classes.values()],axis=-1)

        return new_mask

    @staticmethod
    def resize(image,target_size):
        image = cv.resize(
            image, target_size[::-1],
            cv.INTER_CUBIC
        )
        print(image.shape)
        return image

    def normalise(self, bands, band_ids):
        for i,band_id in enumerate(band_ids):
            data = bands[...,i]
            bm = self.sm['bands'][band_id] # Get a shortcut for the band's netadata

            gain = bm['gain']
            if isinstance(gain, str):
                gain = float(self.tm[gain])
            offset = bm['offset']
            if isinstance(offset, str):
                offset = float(self.tm[offset])
            data = data * gain + offset

            if bm['type'] == 'TOA Normalised Brightness Temperature':
                data = (bm['K2']  / np.log(bm['K2'] / data + 1))
                data = (data - bm['MINIMUM_BT']) / (bm['MAXIMUM_BT'] - bm['MINIMUM_BT'])

            if bm.get('solar_correction', False):
                data /= math.sin(float(self.tm['SUN_ELEVATION'])*math.pi/180)

            bands[...,i] = data
        return bands

class L7Irish206(Dataset):
    def __init__(self, **kwargs):
        super().__init__(satellite_id='Landsat7', **kwargs)
        #PLACEHOLDER FOR classes.yaml file. Need solution to Irish labelling problem.
        self.classes = {'FILL': 0, 'SHADOW':1, 'CLEAR':2,'THIN CLOUD': 3, 'THICK CLOUD': 4}
    def get_tm(self, tile_name):
        filename = glob(
            join(abspath(dirname(__file__)), 'constants/datasets/Landsat7_Irish206', tile_name, '*_MTL.txt')
        )
        if not filename:
            return None
        filename = filename[0]
        with open(filename) as file:
            return {
                entry[0]: entry[1]
                for entry in map(lambda l: "".join(l.split()).split('='), file)
                if len(entry) == 2
            }

    def process(self):
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

        # The user might want to have only some bands:
        if self.bands is not None:
            required_bands = {
                old: new
                for old, new in required_bands.items()
                if new in self.bands
            }

        # We need a fixed order of the bands to compose them later in a numpy array
        required_bands = OrderedDict(sorted(required_bands.items()))

        descriptors = self.get_descriptors(required_bands.values())
        metadata = self.get_metadata(list(required_bands.values()))
        metadata['named_band'] = {
            'RED': required_bands['B30'],
            'GREEN': required_bands['B20'],
            'BLUE': required_bands['B10']
        }

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

            # Get biome and tile id:
            tile_name = join(*directory.split(os.path.sep)[-2:])

            # Get current tile metadata:
            self.tm = self.get_tm(tile_name)
            if self.tm is None:
                print('Tile skipped since no tile metadata was found!')
                continue

            metadata['sun_elevation'] = self.tm['SUN_ELEVATION']

            # We need the bands in a sorted ascending order:
            band_files = sorted(band_files)

            # Retrieve the ids from all bands we need:
            ids, band_files = zip(*[
                [required_bands[file.split('_')[-1][:-4]], file]
                for file in band_files
                if file.split('_')[-1][:-4] in required_bands
            ])
            ids = list(ids)

            # Sanity-check: are the bands from the directory, really the ones that we expect?
            if ids != list(required_bands.values()):
                print(directory, 'was skipped since not all required bands were found!')
                print('Missing bands:', set(required_bands.values()) - set(ids))
                continue

            try:
                bands, no_data = self.load_bands(ids, band_files)
            except ReadingError as err:
                print(err)
                continue

            mask = self.load_mask(mask_file, no_data)

            self.split_and_save(tile_name, bands, mask, no_data, metadata, descriptors)

    def load_mask(self, file, no_data):
        mask = imread(file)
        mask = self.resize_mask(mask, no_data.shape[:2])

        new_mask = np.empty((*mask.shape[:2], 5), dtype=bool)
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

        return new_mask

    def normalise(self, band_with_id):
        band_id, data = band_with_id
        bm = self.sm['bands'][band_id] # Get a shortcut for the band's netadata

        gain = bm['gain']
        if isinstance(gain, str):
            gain = float(self.tm[gain])
        offset = bm['offset']
        if isinstance(offset, str):
            offset = float(self.tm[offset])

        data = data * gain + offset

        if bm['type'] == 'TOA Normalised Brightness Temperature':
            data = (bm['K2']  / np.log(bm['K2'] / data + 1))
            data = (data - bm['MINIMUM_BT']) / (bm['MAXIMUM_BT'] - bm['MINIMUM_BT'])

        if bm.get('solar_correction', False):
            data /= math.sin(float(self.tm['SUN_ELEVATION'])*math.pi/180)

        return data

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
        'L8SPARCS80': L8SPARCS80,
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
                            default=0.5)
    arg_parser.add_argument('-b', '--bands', type=str,
                            help='List of bands to be used', default='None')
    arg_parser.add_argument('-s', '--stride', type=int, default=None,
                            help = 'Stride used for patch extraction. If this is smaller than patch_size, '
                                   'patches overlap each other. Default: set to patch_size, so no patches '
                                   'overlap each other.')
    arg_parser.add_argument('-g', '--generate_metadata', type=bool,
                            help='Whether to generate a metadata file for each image/mask pair', default=True)
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
