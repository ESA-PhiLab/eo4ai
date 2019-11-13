import os


class FileFinderBySubStrings:
    """File finder for individual files based on substrings.

    Attributes
    ----------
    root_path : str
        Path to directory that will be searched within.
    """
    def __init__(self, root_path):
        self.root_path = root_path

    def __call__(self, substrings, startswith=None,
                 endswith=None, dir_substrings=None):
        """Finds unique file based on substrings and other conditions.

        Parameters
        ----------
        substrings : list or str
            All strings that file must contain---not including directories or
            self.root_path.
        startswith : str, optional
            String that file must start with---not including directories or
            self.root_path (default is None).
        endswith : str, optional
            String that file must end with (default is None).
        dir_substrings : list or str, optional
            All strings that the directory-tree of file must contain (default
            is None).

        Returns
        -------
        str
            Path to unique file meeting conditions.
        """
        if isinstance(substrings, str):
            substrings = [substrings]
        if isinstance(dir_substrings, str):
            dir_substrings = [dir_substrings]

        found_paths = []
        for root, dirs, paths in os.walk(self.root_path):

            for possible_path in paths:
                if startswith is not None:
                    if not possible_path.startswith(startswith):
                        continue
                if endswith is not None:
                    if not possible_path.endswith(endswith):
                        continue
                if dir_substrings is not None:
                    if not all(
                            substring in root
                            for substring in dir_substrings
                            ):
                        continue
                if all(substring in possible_path for substring in substrings):
                    possible_path = os.path.join(root, possible_path)
                    found_paths.append(possible_path)
        if len(found_paths) == 0:
            raise Exception('Failed to find any files matching criteria!')
        if len(found_paths) > 1:
            raise Exception('Found multiple files matching criteria!')

        return found_paths[0]


class BandRegisterFinder:
    """File finder for multiple files based on substrings.

    Attributes
    ----------
    dataset_metadata : dict
        Dataset-specific values and information.
    root_path : str
        Path to directory that will be searched within.
    filefinder : FileFinderBySubStrings
        Finder for single files used to construct register.
    """
    def __init__(self, dataset_metadata, root_path):
        self.dataset_metadata = dataset_metadata
        self.root_path = root_path
        self.filefinder = FileFinderBySubStrings(self.root_path)

    def __call__(self, substrings='', startswith=None,
                 endswith=None, dir_substrings=None):
        """Finds required files based on dataset_metadata, substrings, and
        other conditions.

        Parameters
        ----------
        substrings : list or str, optional
            All strings that all files must contain---not including directories
            or self.root_path (default is '').
        startswith : str, optional
            String that all files must start with---not including directories
            or self.root_path (default is None).
        endswith : str, optional
            String that all files must end with (default is None).
        dir_substrings : list or str, optional
            All strings that the directory-tree of all files must contain
            (default is None).

        Returns
        -------
        band_file_register : dict
            Dictionary mapping paths to the band data they contain as lists of
            band_ids.
        """
        if isinstance(substrings, str):
            substrings = [substrings]

        band_file_register = {}
        for band_file_substrings, band_file_ids in self.dataset_metadata[
                                                    'band_files'
                                                    ].items():

            if isinstance(band_file_substrings, str):
                band_file_substrings = [band_file_substrings]

            path = self.filefinder(
                                band_file_substrings + substrings,
                                startswith=startswith,
                                endswith=endswith,
                                dir_substrings=dir_substrings
                                )
            band_file_register[path] = [band_file_ids]

        return band_file_register
