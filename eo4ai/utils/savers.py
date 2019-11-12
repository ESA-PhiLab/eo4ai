from abc import abstractmethod, ABC
import json
import os
import numpy as np


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
                np.save(image_path,image.astype(np.float32))
            if not os.exists(mask_path):
                np.save(mask_path,mask.astype(bool))
        else:
            np.save(image_path,image.astype(np.float32))
            np.save(mask_path,mask.astype(bool))

class ImageMaskDescriptorNumpySaver(ImageMaskNumpySaver):
    def __call__(self,images,masks,descriptors,out_paths):
        self._make_dirs(out_paths)
        if not isinstance(descriptors,list):
            descriptors = [descriptors]*len(out_paths)
        for item in zip(images,masks,descriptors,out_paths):
            self._save_sample(*item)
    def _save_sample(self,image,mask,descriptors,out_path):
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
