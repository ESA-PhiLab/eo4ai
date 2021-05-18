from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
import getpass
import glymur
import json
import numpy as np
import os
from sentinelsat import SentinelAPI
import spectral as spy
import tifffile as tif
import traceback
import yaml
from zipfile import ZipFile

from eo4ai.utils import encoders, filefinders, filters, loaders, misc
from eo4ai.utils import normalisers, resizers, savers, splitters, writers


class Dataset(ABC):
    """Base class for datasets.
    """
    def __init__(self, dataset_id, jobs, **kwargs):
        self.dataset_id = dataset_id
        self.jobs = jobs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.metadata_file = self.get_dataset_metadata()

    def get_dataset_metadata(self):
        metadata_file = os.path.join(
                                os.path.abspath(os.path.dirname(__file__)),
                                '..',
                                'constants',
                                'datasets',
                                self.dataset_id,
                                self.dataset_id + '.yaml'
                                )
        with open(metadata_file, 'r') as stream:
            try:
                self.dataset_metadata = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc

    def process(self):
        scenes = self.get_scenes()
        if self.jobs < 2:
            for scene in scenes:
                self._save_process_scene(scene)
        else:
            with ProcessPoolExecutor(self.jobs) as pool:
                pool.map(self._save_process_scene, scenes)
        # self.dump_README() # TODO

    def _save_process_scene(self, scene):
        print('PROCESSING:', scene)
        try:
            self.process_scene(scene)
        except Exception as err:
            print(
                 'ERROR while processing',
                 scene,
                 str(err),
                 ''.join(traceback.format_tb(err.__traceback__))
                 )

    def dump_README(self):
        # TODO
        pass

    @abstractmethod
    def get_scenes(self):
        pass

    @abstractmethod
    def process_scene(self):
        pass


class L8SPARCS80(Dataset):

    def __init__(self, **kwargs):
        super().__init__(dataset_id='L8SPARCS80', **kwargs)
        self.filefinder = filefinders.FileFinderBySubStrings(self.in_path)
        self.bandloader = loaders.SingleFileBandLoader(self.dataset_metadata)
        self.maskloader = loaders.ImageLoader(self.dataset_metadata)
        self.metadataloader = loaders.LandsatMTLLoader()
        self.normaliser = normalisers.Landsat8Normaliser(self.dataset_metadata)
        self.encoder = encoders.MapByColourEncoder(self.dataset_metadata)
        self.descriptorloader = loaders.SimpleSpectralDescriptorsLoader(
                                                        self.dataset_metadata
                                                        )
        self.descriptors = self.descriptorloader(
                                                band_ids=self.selected_band_ids
                                                )
        self.resizer = resizers.BandsMaskResizer(
                                                self.dataset_metadata,
                                                to_array=True
                                                )
        self.splitter = splitters.SlidingWindowSplitter(
                                                    self.patch_size,
                                                    self.stride
                                                    )
        self.outputmetadatawriter = writers.LandsatMetadataWriter(
                                                        self.dataset_metadata,
                                                        sun_elevation=True
                                                        )
        self.outputorganiser = misc.BySceneAndPatchOrganiser()
        self.datasaver = savers.ImageMaskDescriptorNumpySaver(overwrite=True)
        self.metadatasaver = savers.MetadataJsonSaver(overwrite=True)

    def get_scenes(self):
        return list(set([file[:21] for file in os.listdir(self.in_path)]))

    def process_scene(self, scene_id):
        # Find scene's files
        band_file = self.filefinder('_data', startswith=scene_id)
        mask_file = self.filefinder(
                                self.dataset_metadata['mask']['mask_file'],
                                startswith=scene_id
                                )
        scene_metadata_file = self.filefinder('_mtl', startswith=scene_id)

        # Load bands, mask and metadata
        bands, band_ids = self.bandloader(
                                    band_file,
                                    '_data',
                                    selected_band_ids=self.selected_band_ids
                                    )
        # TODO: Horrible hack to fix RGBA coming from different OSs
        mask = self.maskloader(mask_file)[..., :3]
        self.scene_metadata = self.metadataloader(scene_metadata_file)

        # Normalise band values
        bands = self.normaliser(
                                bands,
                                band_ids,
                                self.scene_metadata,
                                nodata_as=0
                                )

        # Encode mask
        mask, class_ids = self.encoder(mask)

        # Resize bands and mask
        bands, mask = self.resizer(bands, band_ids, mask, self.resolution)

        # Split into patches
        band_patches, mask_patches, patch_ids = self.splitter(bands, mask)

        # Get output_metadata
        self.output_metadata = self.outputmetadatawriter(
                scene_id, band_ids, class_ids,
                scene_metadata=self.scene_metadata, resolution=self.resolution
                )
        # Get directories for outputs
        output_paths = self.outputorganiser(self.out_path, scene_id, patch_ids)

        # Save data
        self.datasaver(
            band_patches, mask_patches, self.descriptors, output_paths
            )

        # Save metadata
        self.metadatasaver(self.output_metadata, output_paths)


class L8Biome96(Dataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_id='L8Biome96', **kwargs)
        self.bandregisterfinder = filefinders.BandRegisterFinder(
                                                        self.dataset_metadata,
                                                        self.in_path
                                                        )
        self.filefinder = filefinders.FileFinderBySubStrings(self.in_path)
        self.bandloader = loaders.MultiFileBandLoader(
                                                        self.dataset_metadata,
                                                        imread=tif.imread
                                                        )
        self.maskloader = loaders.ImageLoader(
                                                    self.dataset_metadata,
                                                    imread=self._mask_imread
                                                    )
        self.metadataloader = loaders.LandsatMTLLoader()
        self.normaliser = normalisers.Landsat8Normaliser(self.dataset_metadata)
        self.encoder = encoders.MapByValueEncoder(self.dataset_metadata)
        self.descriptorloader = loaders.SimpleSpectralDescriptorsLoader(
                                                        self.dataset_metadata
                                                        )
        self.descriptors = self.descriptorloader(
                                                band_ids=self.selected_band_ids
                                                )
        self.resizer = resizers.BandsMaskResizer(
                            self.dataset_metadata, to_array=True, strict=False
                            )
        self.splitter = splitters.SlidingWindowSplitter(
                                        self.patch_size,
                                        self.stride,
                                        filters=[filters.FilterByMaskClass(
                                            threshold=self.nodata_threshold,
                                            target_index=0
                                            )]
                                        )
        self.outputmetadatawriter = writers.LandsatMetadataWriter(
                                                        self.dataset_metadata,
                                                        sun_elevation=True
                                                        )
        self.outputorganiser = misc.BySceneAndPatchOrganiser()
        self.datasaver = savers.ImageMaskDescriptorNumpySaver(overwrite=True)
        self.metadatasaver = savers.MetadataJsonSaver(overwrite=True)

    @staticmethod
    def _mask_imread(filename):
        return np.squeeze(spy.open_image(filename).load())

    def get_scenes(self):
        scenes = []
        for root, dirs, paths in os.walk(self.in_path):
            if any([
                    '_MTL' in path for path in paths
                    ]) and any([
                    path.lower().endswith('_fixedmask.hdr')
                    for path in paths
                    ]):
                scenes.append(root.replace(
                                self.in_path + os.sep,
                                ''
                                ).replace(self.in_path, ''))
        return scenes

    def process_scene(self, scene_id):
        # Find scene's files
        band_file_register = self.bandregisterfinder(dir_substrings=scene_id)
        mask_file = self.filefinder(
                                self.dataset_metadata['mask']['mask_file'],
                                dir_substrings=scene_id
                                )
        scene_metadata_file = self.filefinder('_MTL', dir_substrings=scene_id)
        # Load bands, mask and metadata
        bands, band_ids = self.bandloader(
                                    band_file_register,
                                    selected_band_ids=self.selected_band_ids
                                    )
        mask = self.maskloader(mask_file)
        self.scene_metadata = self.metadataloader(scene_metadata_file)
        # Normalise band values
        bands = self.normaliser(
                            bands, band_ids, self.scene_metadata, nodata_as=0
                            )

        # Encode mask
        mask, class_ids = self.encoder(mask)

        # Resize bands and mask
        bands, mask = self.resizer(bands, band_ids, mask, self.resolution)

        # Split into patches
        band_patches, mask_patches, patch_ids = self.splitter(bands, mask)

        # Get output_metadata
        self.output_metadata = self.outputmetadatawriter(
                                            scene_id, band_ids, class_ids,
                                            scene_metadata=self.scene_metadata,
                                            resolution=self.resolution
                                            )
        # Get directories for outputs
        output_paths = self.outputorganiser(self.out_path, scene_id, patch_ids)
        # Save data
        self.datasaver(
                    band_patches, mask_patches, self.descriptors, output_paths
                    )

        # Save metadata
        self.metadatasaver(self.output_metadata, output_paths)


class L7Irish206(Dataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_id='L7Irish206', **kwargs)
        self.bandregisterfinder = filefinders.BandRegisterFinder(
                                            self.dataset_metadata,
                                            self.in_path
                                            )
        self.filefinder = filefinders.FileFinderBySubStrings(self.in_path)
        self.bandloader = loaders.MultiFileBandLoader(
                                        self.dataset_metadata,
                                        imread=tif.imread
                                        )
        self.maskloader = loaders.ImageLoader(
                                        self.dataset_metadata,
                                        imread=tif.imread
                                        )
        self.metadataloader = loaders.LandsatMTLLoader()
        self.normaliser = normalisers.Landsat7Pre2011Normaliser(
                                                        self.dataset_metadata
                                                        )
        self.encoder = encoders.L7IrishEncoder(self.dataset_metadata)
        self.descriptorloader = loaders.SimpleSpectralDescriptorsLoader(
                                                        self.dataset_metadata
                                                        )
        self.descriptors = self.descriptorloader(
                                            band_ids=self.selected_band_ids
                                            )
        self.resizer = resizers.BandsMaskResizer(
                                                self.dataset_metadata,
                                                to_array=True,
                                                strict=False
                                                )
        self.splitter = splitters.SlidingWindowSplitter(
                                        self.patch_size,
                                        self.stride,
                                        filters=[filters.FilterByMaskClass(
                                            threshold=self.nodata_threshold,
                                            target_index=0
                                            )]
                                        )
        self.outputmetadatawriter = writers.LandsatMetadataWriter(
                                                        self.dataset_metadata,
                                                        sun_elevation=True
                                                        )
        self.outputorganiser = misc.BySceneAndPatchOrganiser()
        self.datasaver = savers.ImageMaskDescriptorNumpySaver(overwrite=True)
        self.metadatasaver = savers.MetadataJsonSaver(overwrite=True)

    def get_scenes(self):
        scenes = []
        for root, dirs, paths in os.walk(self.in_path):
            if (any(
                    ['_MTL' in path for path in paths]
                    )
                    and any([
                            path.lower().endswith('mask2019.tif')
                            for path in paths
                            ])):
                # TODO: Investigate alternative patterns
                scenes.append(root.replace(
                                        self.in_path + os.sep,
                                        ''
                                        ).replace(self.in_path, ''))
        return scenes

    def process_scene(self, scene_id):
        # Find scene's files
        band_file_register = self.bandregisterfinder(dir_substrings=scene_id)
        mask_file = self.filefinder(
                                    self.dataset_metadata['mask']['mask_file'],
                                    dir_substrings=scene_id
                                    )
        scene_metadata_file = self.filefinder('_MTL', dir_substrings=scene_id)

        # Load bands, mask and metadata
        bands, band_ids = self.bandloader(
                                    band_file_register,
                                    selected_band_ids=self.selected_band_ids
                                    )
        mask = self.maskloader(mask_file)
        self.scene_metadata = self.metadataloader(scene_metadata_file)

        # Encode mask
        mask, class_ids = self.encoder(mask, bands)

        # Normalise band values
        bands = self.normaliser(
                            bands,
                            band_ids,
                            self.scene_metadata,
                            nodata_as=0
                            )

        # Resize bands and mask
        bands, mask = self.resizer(bands, band_ids, mask, self.resolution)

        # Split into patches
        band_patches, mask_patches, patch_ids = self.splitter(bands, mask)

        # Get output_metadata
        self.output_metadata = self.outputmetadatawriter(
                                            scene_id, band_ids, class_ids,
                                            scene_metadata=self.scene_metadata,
                                            resolution=self.resolution
                                            )
        # Get directories for outputs
        output_paths = self.outputorganiser(self.out_path, scene_id, patch_ids)
        # Save data
        self.datasaver(
                    band_patches,
                    mask_patches,
                    self.descriptors,
                    output_paths
                    )

        # Save metadata
        self.metadatasaver(self.output_metadata, output_paths)


class S2CESBIO38(Dataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_id='S2CESBIO38', **kwargs)
        self.bandregisterfinder = filefinders.BandRegisterFinder(
                                                        self.dataset_metadata,
                                                        self.in_path
                                                        )
        self.filefinder = filefinders.FileFinderBySubStrings(self.in_path)
        self.bandloader = loaders.MultiFileBandLoader(
                                                    self.dataset_metadata,
                                                    imread=self._band_imread
                                                    )
        self.maskloader = loaders.ImageLoader(
                                            self.dataset_metadata,
                                            imread=tif.imread
                                            )
        self.normaliser = normalisers.Landsat8Normaliser(self.dataset_metadata)
        self.encoder = encoders.MapByValueEncoder(self.dataset_metadata)
        self.descriptorloader = loaders.SimpleSpectralDescriptorsLoader(
                                                        self.dataset_metadata
                                                        )
        self.descriptors = self.descriptorloader(
                                                band_ids=self.selected_band_ids
                                                )
        self.resizer = resizers.BandsMaskResizer(
                                                self.dataset_metadata,
                                                to_array=True,
                                                strict=False
                                                )
        self.splitter = splitters.SlidingWindowSplitter(
                                        self.patch_size,
                                        self.stride,
                                        filters=[filters.FilterByMaskClass(
                                            threshold=self.nodata_threshold,
                                            target_index=0
                                            )]
                                        )
        self.outputmetadatawriter = writers.LandsatMetadataWriter(
                                                        self.dataset_metadata,
                                                        sun_elevation=False
                                                        )
        self.outputorganiser = misc.BySceneAndPatchOrganiser()
        self.datasaver = savers.ImageMaskDescriptorNumpySaver(overwrite=True)
        self.metadatasaver = savers.MetadataJsonSaver(overwrite=True)
        self.sensat_username = None
        self.sensat_passwd = None
        self.download_scenes()

    @staticmethod
    def _band_imread(filename):
        return glymur.Jp2k(filename)[:]

    def download_present(self, scene_id):
        for root, dirs, paths in os.walk(scene_id):
            for dir in dirs:
                if dir.endswith('IMG_DATA') or dir.endswith('IMG_DATA/'):
                    return True
        return False

    def download_scenes(self):
        scene_ids = self.get_scenes(only_downloaded=False)
        with open(os.path.join(
                            os.path.abspath(os.path.dirname(__file__)),
                            '..',
                            'constants',
                            'datasets',
                            'S2CESBIO38',
                            'sceneIDs.yaml'
                            ), 'r') as f:
            try:
                self.product_id_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise exc
        for scene in scene_ids:
            if not self.download_present(os.path.join(self.in_path, scene)):
                self.download_scene(scene)

    def download_scene(self, scene_id):
        scene_dir = os.path.join(self.in_path, scene_id)
        with open(os.path.join(scene_dir, 'used_parameters.json'), 'r') as f:
            scene_parameters = json.load(f)
        original_cloudy_product_id = scene_parameters['cloudy_product_name']
        downloadable_product_id = self.product_id_dict[
                                                    original_cloudy_product_id
                                                    ]
        if self.sensat_username is None:
            self.sensat_username = input('Please enter SentinelHub username: ')
            self.sensat_passwd = getpass.getpass(
                                        'Please enter SentinelHub password: '
                                        )
            self.api = SentinelAPI(self.sensat_username, self.sensat_passwd)
        prod = self.api.query(raw=downloadable_product_id)
        self.api.download_all(prod, directory_path=scene_dir)

        os.makedirs(os.path.join(scene_dir, 'IMG_DATA'))
        with ZipFile(os.path.join(
                                scene_dir,
                                downloadable_product_id + '.zip'
                                ), 'r') as zf:
            for file in zf.namelist():
                if ('IMG_DATA' in file and not (
                            file.endswith('IMG_DATA')
                            or file.endswith('IMG_DATA/')
                            )):
                    file_data = zf.read(file)
                    with open(os.path.join(
                                        scene_dir,
                                        'IMG_DATA',
                                        os.path.basename(file)
                                        ), "wb") as fout:
                        fout.write(file_data)

        os.remove(os.path.join(scene_dir, downloadable_product_id + '.zip'))

    def get_scenes(self, only_downloaded=True):
        scenes = []
        for root, dirs, paths in os.walk(self.in_path):
            if any(['classification_map' in path for path in paths]):
                if only_downloaded and self.download_present(root):
                    scenes.append(os.path.dirname(
                        root.replace(self.in_path + os.sep, '').replace(
                                                            self.in_path, ''
                                                            )
                        ))
                elif not only_downloaded:
                    scenes.append(os.path.dirname(
                        root.replace(self.in_path + os.sep, '').replace(
                                                            self.in_path, ''
                                                            )
                        ))
        return scenes

    def process_scene(self, scene_id):
        # Find scene's files
        band_file_register = self.bandregisterfinder(dir_substrings=[scene_id])
        mask_file = self.filefinder(
                                self.dataset_metadata['mask']['mask_file'],
                                dir_substrings=scene_id
                                )
        # Load bands, mask and metadata
        bands, band_ids = self.bandloader(
                                    band_file_register,
                                    selected_band_ids=self.selected_band_ids
                                    )
        mask = self.maskloader(mask_file)
        # Normalise band values
        bands = self.normaliser(bands, band_ids, None, nodata_as=0)

        # Encode mask
        mask, class_ids = self.encoder(mask)

        # Resize bands and mask
        bands, mask = self.resizer(bands, band_ids, mask, self.resolution)

        # Split into patches
        band_patches, mask_patches, patch_ids = self.splitter(bands, mask)

        # Get output_metadata
        self.output_metadata = self.outputmetadatawriter(
                                                    scene_id,
                                                    band_ids,
                                                    class_ids,
                                                    resolution=self.resolution
                                                    )
        # Get directories for outputs
        output_paths = self.outputorganiser(self.out_path, scene_id, patch_ids)
        # Save data
        self.datasaver(
                    band_patches,
                    mask_patches,
                    self.descriptors,
                    output_paths
                    )

        # Save metadata
        self.metadatasaver(self.output_metadata, output_paths)


class S2IRIS513(Dataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_id='S2IRIS513', **kwargs)

        self.filefinder = filefinders.FileFinderBySubStrings(self.in_path)
        self.bandloader = loaders.SingleFileBandLoader(
                                                    self.dataset_metadata,
                                                    imread=np.load
                                                    )
        self.maskloader = loaders.ImageLoader(
                                            self.dataset_metadata,
                                            imread=np.load
                                            )

        self.descriptorloader = loaders.SimpleSpectralDescriptorsLoader(
                                                        self.dataset_metadata
                                                        )
        self.descriptors = self.descriptorloader(
                                                band_ids=self.selected_band_ids
                                                )

        self.resizer = resizers.BandsMaskResizer(
                                                self.dataset_metadata,
                                                to_array=True,
                                                strict=False
                                                )

        self.splitter = splitters.SlidingWindowSplitter(
                                        self.patch_size,
                                        self.stride,
                                        filters=[filters.FilterByMaskClass(
                                            threshold=self.nodata_threshold,
                                            target_index=0
                                            )]
                                        )

        self.outputmetadatawriter = writers.LandsatMetadataWriter(
                                                        self.dataset_metadata,
                                                        sun_elevation=False
                                                        )
        self.outputorganiser = misc.BySceneAndPatchOrganiser()
        self.datasaver = savers.ImageMaskDescriptorNumpySaver(overwrite=True)
        self.metadatasaver = savers.MetadataJsonSaver(overwrite=True)

    def get_scenes(self):
        image_ids = os.listdir(os.path.join(self.in_path, 'subscenes'))
        mask_ids = os.listdir(os.path.join(self.in_path, 'masks'))
        assert set(image_ids) == set(mask_ids), 'Different scenes in images' \
                                                'and masks!'

        return [id.replace('.npy', '') for id in image_ids]

    def process_scene(self, scene_id):
        # Find scene's files
        band_file = self.filefinder(scene_id, dir_substrings='subscenes')
        mask_file = self.filefinder(scene_id, dir_substrings='masks')
        # Load bands, mask and metadata
        bands, band_ids = self.bandloader(
                            band_file,
                            list(self.dataset_metadata['band_files'].keys())[0]
                            )

        mask = self.maskloader(mask_file)
        class_ids = list(self.dataset_metadata['mask']['classes'].keys())

        # Resize bands and mask
        bands, mask = self.resizer(bands, band_ids, mask, self.resolution)

        # Split into patches
        band_patches, mask_patches, patch_ids = self.splitter(bands, mask)

        # Get output_metadata
        self.output_metadata = self.outputmetadatawriter(
                                                    scene_id,
                                                    band_ids,
                                                    class_ids,
                                                    resolution=self.resolution
                                                    )
        # Get directories for outputs
        output_paths = self.outputorganiser(self.out_path, scene_id, patch_ids)
        # Save data
        self.datasaver(
                    band_patches,
                    mask_patches,
                    self.descriptors,
                    output_paths
                    )

        # Save metadata
        self.metadatasaver(self.output_metadata, output_paths)
