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

import utils

import pprint
import types

class ReadingError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super(ReadingError, self).__init__(message)


class Dataset(ABC):
    """Base class for datasets.
    """
    def __init__(self, dataset_id, in_path, generate_metadata, patch_size, stride, jobs, resolution, out_path, nodata_threshold, selected_band_ids):
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

        # We have two metadata variables: dataset and scene metadata.
        # scene_metadata should change for each new tile.
        self.dataset_metadata = self.get_dataset_metadata(dataset_id)
        self.scene_metadata = None

        # Convert band ids into a list:
        self.selected_band_ids = literal_eval(selected_band_ids)

        # TODO
        self.README_config = locals()


    def process(self):
        scenes = self.get_scenes()
        # with ThreadPoolExecutor(self.jobs) as pool:
        #     pool.map(self.process_scene, scenes)
        for scene in scenes:
            self.process_scene(scene)
        self.dump_README() # TODO

    def dump_README(self):
        # TODO
        pass

    @abstractmethod
    def get_scenes(self):
        pass

    @abstractmethod
    def process_scene(self):
        pass

    @staticmethod
    def get_dataset_metadata(dataset_id):
        with open(join(abspath(dirname(__file__)), 'constants','datasets',dataset_id, dataset_id + '.yaml'), 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc

class L8SPARCS80(Dataset):

    def __init__(self,**kwargs):
        super().__init__(dataset_id='L8SPARCS80', **kwargs)
        self.filefinder = utils.FileFinderBySubStrings(self.in_path)
        self.bandloader = utils.SingleFileBandLoader(self.dataset_metadata)
        self.maskloader = utils.ImageLoader()
        self.metadataloader = utils.LandsatMTLLoader()
        self.normaliser = utils.Landsat8Normaliser(self.dataset_metadata)
        self.encoder = utils.MapByColourEncoder(self.dataset_metadata)
        self.descriptorloader = utils.SimpleSpectralDescriptorsLoader(self.dataset_metadata)
        self.descriptors = self.descriptorloader(band_ids=self.selected_band_ids)
        self.resizer = utils.BandsMaskResizer(self.dataset_metadata,to_array=True)
        self.splitter = utils.SlidingWindowSplitter(self.patch_size,self.stride)
        self.outputmetadatawriter = utils.LandsatMetadataWriter(self.dataset_metadata,sun_elevation=True)
        self.outputorganiser = utils.BySceneAndPatchOrganiser()
        self.datasaver = utils.ImageMaskDescriptorNumpySaver(overwrite=True)
        self.metadatasaver = utils.MetadataJsonSaver(overwrite=True)

    def get_scenes(self):
        return list(set([file[:21] for file in os.listdir(self.in_path)]))

    def process_scene(self,scene_id):
        #Find scene's files
        band_file = self.filefinder('_data',startswith=scene_id)
        mask_file = self.filefinder(self.dataset_metadata['mask']['mask_file'],startswith=scene_id)
        scene_metadata_file = self.filefinder('_mtl',startswith=scene_id)

        #Load bands, mask and metadata
        bands,band_ids = self.bandloader(band_file,'_data',selected_band_ids=self.selected_band_ids)
        mask = self.maskloader(mask_file)
        self.scene_metadata = self.metadataloader(scene_metadata_file)

        #Normalise band values
        bands = self.normaliser(bands,band_ids,self.scene_metadata)

        #Encode mask
        mask,class_ids = self.encoder(mask)

        #Resize bands and mask
        bands,mask = self.resizer(bands,band_ids,mask,self.resolution)

        #Split into patches
        band_patches, mask_patches, patch_ids = self.splitter(bands,mask)

        #get output_metadata
        self.output_metadata = self.outputmetadatawriter(self.scene_metadata,scene_id,band_ids,class_ids,resolution=self.resolution)
        #Get directories for outputs
        output_paths = self.outputorganiser(self.out_path,scene_id,patch_ids)

        #Save data
        self.datasaver(band_patches,mask_patches,self.descriptors,output_paths)

        #Save metadata
        self.metadatasaver(self.output_metadata,output_paths)


class L8Biome96(Dataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_id='L8Biome96', **kwargs)
        self.bandregisterfinder = utils.BandRegisterFinder(self.dataset_metadata,self.in_path)
        self.filefinder = utils.FileFinderBySubStrings(self.in_path)
        self.bandloader = utils.MultiFileBandLoader(self.dataset_metadata,imread=tif.imread)
        self.maskloader = utils.ImageLoader(imread='np.squeeze(spy.open_image("{}").load())')
        self.metadataloader = utils.LandsatMTLLoader()
        self.normaliser = utils.Landsat8Normaliser(self.dataset_metadata)
        self.encoder = utils.MapByValueEncoder(self.dataset_metadata)
        self.descriptorloader = utils.SimpleSpectralDescriptorsLoader(self.dataset_metadata)
        self.descriptors = self.descriptorloader(band_ids=self.selected_band_ids)
        self.resizer = utils.BandsMaskResizer(self.dataset_metadata,to_array=True,strict=False)
        self.splitter = utils.SlidingWindowSplitter(self.patch_size,self.stride,filters=[utils.FilterByMaskClass(threshold = self.nodata_threshold,target_index=0)])
        self.outputmetadatawriter = utils.LandsatMetadataWriter(self.dataset_metadata,sun_elevation=True)
        self.outputorganiser = utils.BySceneAndPatchOrganiser()
        self.datasaver = utils.ImageMaskDescriptorNumpySaver(overwrite=True)
        self.metadatasaver = utils.MetadataJsonSaver(overwrite=True)
    def get_scenes(self):
        scenes = []
        for root,dirs,paths in os.walk(self.in_path):
            if any(['_MTL' in path for path in paths]) and any([path.lower().endswith('_fixedmask.hdr') for path in paths]):
                scenes.append(root.replace(self.in_path+os.sep,''))
        return scenes

    def process_scene(self,scene_id):
        #Find scene's files
        band_file_register = self.bandregisterfinder(dir_substrings=scene_id)
        mask_file = self.filefinder(self.dataset_metadata['mask']['mask_file'],dir_substrings=scene_id)
        scene_metadata_file = self.filefinder('_MTL',dir_substrings=scene_id)
        print(self.out_path)
        pprint.pprint(scene_id)
        #Load bands, mask and metadata
        bands,band_ids = self.bandloader(band_file_register,selected_band_ids=self.selected_band_ids)
        mask = self.maskloader(mask_file)
        self.scene_metadata = self.metadataloader(scene_metadata_file)
        #Normalise band values
        bands = self.normaliser(bands,band_ids,self.scene_metadata)

        #Encode mask
        mask,class_ids = self.encoder(mask)

        #Resize bands and mask
        bands,mask = self.resizer(bands,band_ids,mask,self.resolution)

        #Split into patches
        band_patches, mask_patches, patch_ids = self.splitter(bands,mask)

        #get output_metadata
        self.output_metadata = self.outputmetadatawriter(self.scene_metadata,scene_id,band_ids,class_ids,resolution=self.resolution)
        #Get directories for outputs
        output_paths = self.outputorganiser(self.out_path,scene_id,patch_ids)
        #Save data
        self.datasaver(band_patches,mask_patches,self.descriptors,output_paths)

        #Save metadata
        self.metadatasaver(self.output_metadata,output_paths)


class L7Irish206(Dataset):
    def __init__(self,**kwargs):
        super().__init__(dataset_id='L7Irish206', **kwargs)
        self.bandregisterfinder = utils.BandRegisterFinder(self.dataset_metadata,self.in_path)
        self.filefinder = utils.FileFinderBySubStrings(self.in_path)
        self.bandloader = utils.MultiFileBandLoader(self.dataset_metadata,imread=tif.imread)
        self.maskloader = utils.ImageLoader(imread=tif.imread)
        self.metadataloader = utils.LandsatMTLLoader()
        self.normaliser = utils.Landsat7Pre2011Normaliser(self.dataset_metadata)
        self.encoder = utils.L7IrishEncoder(self.dataset_metadata)
        self.descriptorloader = utils.SimpleSpectralDescriptorsLoader(self.dataset_metadata)
        self.descriptors = self.descriptorloader(band_ids=self.selected_band_ids)
        self.resizer = utils.BandsMaskResizer(self.dataset_metadata,to_array=True,strict=False)
        self.splitter = utils.SlidingWindowSplitter(self.patch_size,self.stride,filters=[utils.FilterByMaskClass(threshold = self.nodata_threshold,target_index=0)])
        self.outputmetadatawriter = utils.LandsatMetadataWriter(self.dataset_metadata,sun_elevation=True)
        self.outputorganiser = utils.BySceneAndPatchOrganiser()
        self.datasaver = utils.ImageMaskDescriptorNumpySaver(overwrite=True)
        self.metadatasaver = utils.MetadataJsonSaver(overwrite=True)

    def get_scenes(self):
        scenes = []
        for root,dirs,paths in os.walk(self.in_path):
            if any(['_MTL' in path for path in paths]) and any([path.lower().endswith('mask2019.tif') for path in paths]):
                scenes.append(root.replace(self.in_path+os.sep,''))
        return scenes

    def process_scene(self,scene_id):
        #Find scene's files
        band_file_register = self.bandregisterfinder(dir_substrings=scene_id)
        mask_file = self.filefinder(self.dataset_metadata['mask']['mask_file'],dir_substrings=scene_id)
        scene_metadata_file = self.filefinder('_MTL',dir_substrings=scene_id)

        #Load bands, mask and metadata
        bands,band_ids = self.bandloader(band_file_register,selected_band_ids=self.selected_band_ids)
        mask = self.maskloader(mask_file)
        self.scene_metadata = self.metadataloader(scene_metadata_file)

        #Encode mask
        mask,class_ids = self.encoder(mask,bands)

        #Normalise band values
        bands = self.normaliser(bands,band_ids,self.scene_metadata)

        #Resize bands and mask
        bands,mask = self.resizer(bands,band_ids,mask,self.resolution)

        #Split into patches
        band_patches, mask_patches, patch_ids = self.splitter(bands,mask)

        #get output_metadata
        self.output_metadata = self.outputmetadatawriter(self.scene_metadata,scene_id,band_ids,class_ids,resolution=self.resolution)
        #Get directories for outputs
        output_paths = self.outputorganiser(self.out_path,scene_id,patch_ids)
        #Save data
        self.datasaver(band_patches,mask_patches,self.descriptors,output_paths)

        #Save metadata
        self.metadatasaver(self.output_metadata,output_paths)



class L7Irish206_old(Dataset):
    def __init__(self, **kwargs):
        super().__init__(satellite_id='Landsat7', **kwargs)
        #PLACEHOLDER FOR classes.yaml file. Need solution to Irish labelling problem.
        self.classes = {'FILL': 0, 'SHADOW':1, 'CLEAR':2,'THIN CLOUD': 3, 'THICK CLOUD': 4}
    def get_scene_metadata(self, tile_name):
        filename = glob(
            join(abspath(dirname(__file__)), 'constants/datasets/L7Irish206', tile_name, '*_MTL.txt')
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
            self.scene_metadata = self.get_scene_metadata(tile_name)
            if self.scene_metadata is None:
                print('Tile skipped since no tile metadata was found!')
                continue

            metadata['sun_elevation'] = self.scene_metadata['SUN_ELEVATION']

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
            gain = float(self.scene_metadata[gain])
        offset = bm['offset']
        if isinstance(offset, str):
            offset = float(self.scene_metadata[offset])

        data = data * gain + offset

        if bm['type'] == 'TOA Normalised Brightness Temperature':
            data = (bm['K2']  / np.log(bm['K2'] / data + 1))
            data = (data - bm['MINIMUM_BT']) / (bm['MAXIMUM_BT'] - bm['MINIMUM_BT'])

        if bm.get('solar_correction', False):
            data /= math.sin(float(self.scene_metadata['SUN_ELEVATION'])*math.pi/180)

        return data


class S2CESBIO(Dataset):
    def __init__(self, **kwargs):
        pass


if __name__ == '__main__':
    import shutil

    dataset_classes = {
        'L8Biome96': L8Biome96,
        'L8SPARCS80': L8SPARCS80,
        'L7Irish206': L7Irish206,
        'S2CESBIO': S2CESBIO
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
                            default=None)
    arg_parser.add_argument('-b', '--selected_band_ids', type=str,
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


#RUN ME: python prepare_dataset.py  D:\Datasets\clouds\SPARCS_raw .\test_out -d TEST -o True -r 30 -p 256 -g True
