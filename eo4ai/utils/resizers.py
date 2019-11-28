import cv2
import numpy as np


class BandsMaskResizer:
    """Resizer for bands and mask, by changing their resolution.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    to_array : bool, optional
        Whether to convert resized bands into single array (default is True).
    strict : bool, optional
        Whether to disallow resizing operations that lead to bands of different
        sizes (default is True).
    """
    def __init__(self, dataset_metadata, to_array=True, strict=True):
        self.dataset_metadata = dataset_metadata
        self.to_array = to_array
        self.strict = strict

    def __call__(self, bands, band_ids, mask, resolution):
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
        band_target_sizes, mask_target_size = self._get_target_sizes(
                                                            bands,
                                                            band_ids,
                                                            mask,
                                                            resolution
                                                            )
        rescaled_bands = []
        for band, band_id, target_size in zip(
                                            bands,
                                            band_ids,
                                            band_target_sizes
                                            ):
            if not all(np.array(band.shape[:2]) == np.array(target_size)):
                rescaled_bands.append(cv2.resize(
                                            band,
                                            target_size[::-1],
                                            cv2.INTER_CUBIC
                                            ))
            else:
                rescaled_bands.append(band)

        # MASK
        if not all(np.array(mask.shape[:2]) == np.array(mask_target_size)):
            flat_mask = np.argmax(mask, axis=-1)
            rescaled_mask = cv2.resize(
                                    flat_mask.astype(np.uint8),
                                    mask_target_size[::-1],
                                    cv2.INTER_NEAREST
                                    ).astype(bool)
            rescaled_mask = np.stack([
                                    rescaled_mask[:, :] == value
                                    for value in range(mask.shape[-1])
                                    ], axis=-1).astype(np.bool)
        else:
            rescaled_mask = mask

        if self.to_array:
            rescaled_bands = np.moveaxis(np.array(rescaled_bands), 0, -1)
        return rescaled_bands, rescaled_mask

    def _get_target_sizes(self, bands, band_ids, mask, resolution):
        band_scale_factors = [
                    self.dataset_metadata['bands'][b]['resolution']/resolution
                    for b in band_ids
                    ]
        band_target_sizes = [
                    (round(b.shape[0] * s), round(b.shape[1] * s))
                    for b, s in zip(bands, band_scale_factors)
                    ]
        mask_scale_factor \
            = self.dataset_metadata['mask']['resolution']/resolution
        mask_target_size = (
                            round(mask.shape[0] * mask_scale_factor),
                            round(mask.shape[1]*mask_scale_factor)
                            )
        if self.strict and len(set(band_target_sizes)) > 1:
            raise Exception(
                        'Resizing operation leads to inconsistent band'
                        'shapes, which is disallowed when "strict" is True.'
                        )

        elif self.to_array and len(set(band_target_sizes)) > 1:
            most_common = max(
                        set(band_target_sizes),
                        key=band_target_sizes.count
                        )
            band_target_sizes = [most_common] * len(band_target_sizes)
        return band_target_sizes, mask_target_size
