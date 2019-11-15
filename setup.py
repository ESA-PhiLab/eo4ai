from setuptools import setup

setup(
   name='eo4ai',
   description='Tools for preprocessing Earth Observation datasets',
   author='Alistair Francis, John Mrziglod',
   author_email='a.francis.16@ucl.ac.uk',
   packages=['eo4ai', 'eo4ai/utils'],
   license='MIT',
   long_description=open('README.md').read()
)
