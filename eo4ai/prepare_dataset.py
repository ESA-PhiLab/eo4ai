import argparse
from ast import literal_eval
import os
import sys

import datasets


"""Script for transforming original datasets to eo4ai format."""
if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(
        description='Convert raw datasets into eo4ai format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument('dataset',
                            help='Name of dataset class. Check datasets.py for'
                                 'options.')
    arg_parser.add_argument('in_path',
                            help='path to original dataset folder.')
    arg_parser.add_argument('out_path', help='path to output directory')
    arg_parser.add_argument('-o', '--overwrite', action='store_true',
                            help='Allows overwrite of an existing output'
                                 'directory.')
    arg_parser.add_argument('-r', '--resolution', type=float,
                            help='Resolution of output patches (metres)',
                            default=30)
    arg_parser.add_argument('-p', '--patch_size', type=int,
                            help='Size in pixels of each output patch',
                            default=256)
    arg_parser.add_argument('-t', '--nodata_threshold', type=float,
                            help='Fraction of patch that can have no-data'
                                 'values and still be used',
                            default=1)
    arg_parser.add_argument('-b', '--selected_band_ids', type=literal_eval,
                            help='List of bands to be used, defined as list of'
                                 'band_ids.',
                            default=None)
    arg_parser.add_argument('-s', '--stride', type=int,
                            help='Stride used for patch extraction. If this'
                                 'is smaller than patch_size, patches'
                                 'overlap each other. Default: set to'
                                 'patch_size, so no patches overlap each'
                                 'other.',
                            default=256)
    arg_parser.add_argument('-g', '--generate_metadata', action='store_true',
                            help='Whether to generate a metadata file for each'
                                 'image/mask pair')
    # TODO: Implement optional descriptors output
    arg_parser.add_argument('-j', '--jobs', type=int,
                            help='How many parallel jobs should be used to'
                                 'process the data.',
                            default=4)
    #
    #
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
    dataset_class = getattr(datasets, kwargs.pop('dataset'))
    dataset = dataset_class(**kwargs)
    dataset.process()
