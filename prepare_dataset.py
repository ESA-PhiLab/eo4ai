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
import glymur
import json
from zipfile import ZipFile

import utils

import types

from sentinelsat import SentinelAPI

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
        with ThreadPoolExecutor(self.jobs) as pool:
            pool.map(self.process_scene, scenes)
        # for scene in scenes:
            # self.process_scene(scene)
        #self.dump_README() # TODO

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
        self.output_metadata = self.outputmetadatawriter(scene_id,band_ids,class_ids,scene_metadata=self.scene_metadata,resolution=self.resolution)
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
        self.maskloader = utils.ImageLoader(imread=self._mask_imread)
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

    @staticmethod
    def _mask_imread(filename):
        return np.squeeze(spy.open_image(filename).load())

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
        self.output_metadata = self.outputmetadatawriter(scene_id,band_ids,class_ids,scene_metadata=self.scene_metadata,resolution=self.resolution)
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
        self.output_metadata = self.outputmetadatawriter(scene_id,band_ids,class_ids,scene_metadata=self.scene_metadata,resolution=self.resolution)
        #Get directories for outputs
        output_paths = self.outputorganiser(self.out_path,scene_id,patch_ids)
        #Save data
        self.datasaver(band_patches,mask_patches,self.descriptors,output_paths)

        #Save metadata
        self.metadatasaver(self.output_metadata,output_paths)

class S2CESBIO38(Dataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_id='S2CESBIO38', **kwargs)
        self.bandregisterfinder = utils.BandRegisterFinder(self.dataset_metadata,self.in_path)
        self.filefinder = utils.FileFinderBySubStrings(self.in_path)
        self.bandloader = utils.MultiFileBandLoader(self.dataset_metadata,imread=self._band_imread)
        self.maskloader = utils.ImageLoader(imread=tif.imread)
        self.normaliser = utils.Landsat8Normaliser(self.dataset_metadata)
        self.encoder = utils.MapByValueEncoder(self.dataset_metadata)
        self.descriptorloader = utils.SimpleSpectralDescriptorsLoader(self.dataset_metadata)
        self.descriptors = self.descriptorloader(band_ids=self.selected_band_ids)
        self.resizer = utils.BandsMaskResizer(self.dataset_metadata,to_array=True,strict=False)
        self.splitter = utils.SlidingWindowSplitter(self.patch_size,self.stride,filters=[utils.FilterByMaskClass(threshold = self.nodata_threshold,target_index=0)])
        self.outputmetadatawriter = utils.LandsatMetadataWriter(self.dataset_metadata,sun_elevation=False)
        self.outputorganiser = utils.BySceneAndPatchOrganiser()
        self.datasaver = utils.ImageMaskDescriptorNumpySaver(overwrite=True)
        self.metadatasaver = utils.MetadataJsonSaver(overwrite=True)
        self.sensat_username = None
        self.sensat_passwd = None
        self.download_scenes()


    @staticmethod
    def _band_imread(filename):
        return glymur.Jp2k(filename)[:]

    def scene_present(self,scene_id):
        for root,dirs,paths in os.walk(scene_id):
            for dir in dirs:
                if dir.endswith('.SAFE'):
                    return True
        return False

    def download_scenes(self):
        scene_ids = self.get_scenes()
        with open(join(abspath(dirname(__file__)), 'constants','datasets','S2CESBIO38','sceneIDs.yaml'), 'r') as f:
            try:
                self.product_id_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise exc
        for scene in scene_ids:
            if not self.scene_present(scene):
                self.download_scene(scene)

    def download_scene(self,scene_id):
        with open(os.path.join(scene_id,'used_parameters.json'), 'r') as f:
            scene_parameters = json.load(f)
        original_cloudy_product_id = scene_parameters['cloudy_product_name']
        downloadable_product_id = self.product_id_dict[original_cloudy_product_id]
        if self.sensat_username is None:
            self.sensat_username = input('Please enter SentinelHub username: ')
            self.sensat_passwd   = input('Please enter SentinelHub password: ')
            self.api = SentinelAPI(self.sensat_username, self.sensat_passwd)
        prod = self.api.query(raw=downloadable_product_id)
        self.api.download_all(prod,directory_path = scene_id)
        with ZipFile(os.path.join(scene_id,downloadable_product_id+'.zip'),'r') as f:
            f.extractall(scene_id)


    def get_scenes(self):
        scenes = []
        for root,dirs,paths in os.walk(self.in_path):
            if any(['classification_map' in path for path in paths]):
                scenes.append(root.replace(self.in_path+os.sep,''))
        return scenes

    def process_scene(self,scene_id):
        #Find scene's files
        band_file_register = self.bandregisterfinder(dir_substrings=[scene_id])
        mask_file = self.filefinder(self.dataset_metadata['mask']['mask_file'],dir_substrings=scene_id)
        #Load bands, mask and metadata
        bands,band_ids = self.bandloader(band_file_register,selected_band_ids=self.selected_band_ids)
        mask = self.maskloader(mask_file)
        #Normalise band values
        bands = self.normaliser(bands,band_ids,None)

        #Encode mask
        mask,class_ids = self.encoder(mask)

        #Resize bands and mask
        bands,mask = self.resizer(bands,band_ids,mask,self.resolution)

        #Split into patches
        band_patches, mask_patches, patch_ids = self.splitter(bands,mask)

        #get output_metadata
        self.output_metadata = self.outputmetadatawriter(scene_id,band_ids,class_ids,resolution=self.resolution)
        #Get directories for outputs
        output_paths = self.outputorganiser(self.out_path,scene_id,patch_ids)
        #Save data
        self.datasaver(band_patches,mask_patches,self.descriptors,output_paths)

        #Save metadata
        self.metadatasaver(self.output_metadata,output_paths)

if __name__ == '__main__':
    import shutil

    dataset_classes = {
        'L8Biome96': L8Biome96,
        'L8SPARCS80': L8SPARCS80,
        'L7Irish206': L7Irish206,
        'S2CESBIO38': S2CESBIO38
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
