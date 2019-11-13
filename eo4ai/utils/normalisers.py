from abc import abstractmethod, ABC
import math
import numpy as np


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

    def __call__(self,bands,band_ids,scene_metadata,nodata_as=None):
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
            bands[idx] = self._normalise_band(bands[idx],band_id,scene_metadata,nodata_as=nodata_as)
        return bands

    @abstractmethod
    def _normalise_band(self,band,band_id,scene_metadata):
        pass

class Landsat8Normaliser(LandsatNormaliser):
    """Normaliser for Landsat 8 data,to convert to TOA units."""
    def _normalise_band(self,band,band_id,scene_metadata,nodata_as=None):
        """Normalises any Landsat 8 band into TOA units

        Parameters
        ----------
        band : np.ndarray
            Array with Digital Number pixel values from Landsat 8.
        band_id : str
            Identifier for band being normalised.
        scene_metadata : dict
            Scene-specific Landsat metadata values.
        nodata_as : float, optional
            Used to set nodata as given value, left as is if unspecified (default None)


        Returns
        -------
        band : np.ndarray
            Normalised band in TOA units.
        """
        bm = self.dataset_metadata['bands'][band_id] # Get a shortcut for the band's metadata

        gain = bm['gain']
        offset = bm['offset']

        nodata = band == 0
        band = band * gain + offset
        if bm['type'] == 'TOA Normalised Brightness Temperature':
            band = (bm['K2']  / np.log(bm['K1'] / band + 1))
            band = (band - bm['MINIMUM_BT']) / (bm['MAXIMUM_BT'] - bm['MINIMUM_BT'])

        if bm.get('solar_correction', False):
            band /= math.sin(float(self.scene_metadata['SUN_ELEVATION'])*math.pi/180)

        if nodata_as is not None:
            band[nodata] = nodata_as
        return band

class Landsat7Pre2011Normaliser(LandsatNormaliser):
    """Normaliser for Landsat7 Pre-2011 data format, to convert to TOA units."""
    def _normalise_band(self,band,band_id,scene_metadata,nodata_as=None):
        """Normalises any Landsat 7 pre-2011 band into TOA units

        Parameters
        ----------
        band : np.ndarray
            Array with Digital Number pixel values from Landsat 7.
        band_id : str
            Identifier for band being normalised.
        scene_metadata : dict
            Scene-specific Landsat metadata values.
        nodata_as : float, optional
            Used to set nodata as given value, left as is if unspecified (default None)


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
        nodata = band == 0
        radiance = ((L_MAX-L_MIN)/(QCAL_MAX-QCAL_MIN))*(band-QCAL_MIN) + L_MIN

        if bm['type'] == 'TOA Reflectance':
            ESUN = bm['ESUN']
            band = math.pi*radiance/ESUN

            if bm.get('solar_correction', False):
                band /= math.sin(float(self.scene_metadata['SUN_ELEVATION'])*math.pi/180)

        if bm['type'] == 'TOA Normalised Brightness Temperature':
            band = (bm['K2']  / np.log(bm['K1'] / radiance + 1))
            band = (band - bm['MINIMUM_BT']) / (bm['MAXIMUM_BT'] - bm['MINIMUM_BT'])

        if nodata_as is not None:
            band[nodata] = nodata_as

        return band
