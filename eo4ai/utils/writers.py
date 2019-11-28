class LandsatMetadataWriter:
    """Writer for output metadata writing of Landsat datasets.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    sun_elevation : bool, optional
        Whether to include scene-specific value for sun_elevation
        (defaut is False).
    """
    def __init__(self, dataset_metadata, sun_elevation=False):
        self.dataset_metadata = dataset_metadata
        self.sun_elevation = sun_elevation

    def __call__(self, scene_id, band_ids, class_ids,
                 scene_metadata=None, **kwargs):
        """Create dictionary for all relevant metadata values for a Landsat
        scene.

        Parameters
        ----------
        scene_id : str
            Identifier for scene.
        band_ids : list
            Identifiers for bands.
        class_ids : list
            Identifiers for classes.
        scene_metadata : dict, optional
            Scene-specific Landsat metadata values, only needed if
            sun_elevation=True
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
                'bands': band_ids,
                'band_types': [
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
                'classes': class_ids
            }
        if self.sun_elevation:
            output_metadata['sun_elevation'] = scene_metadata['SUN_ELEVATION']

        for band in band_ids:
            descriptive_names = [
                self.dataset_metadata['bands'][band_id].get('descriptive_name')
                for band_id in band_ids
                                ]
            band_positions = [
                    list(self.dataset_metadata['bands'].keys()).index(band_id)
                    for band_id in band_ids
                    ]
            named_bands = {
                          descriptive_name: band_pos
                          for descriptive_name, band_pos
                          in zip(descriptive_names, band_positions)
                          if descriptive_name is not None
                          }

            output_metadata['named_bands'] = named_bands

        # Add any extra user-supplied metadata
        output_metadata.update(kwargs)
        return output_metadata
