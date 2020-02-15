# *---WORK IN PROGRESS---*

# EO4AI

[![Build Status](https://travis-ci.org/ESA-PhiLab/eo4ai.svg?branch=master)](https://travis-ci.org/ESA-PhiLab/eo4ai)

##### *Earth Observation preprocessing tools for AI and machine learning applications*

This project provides easy-to-use tools for preprocessing datasets for image segmentation tasks in Earth Observation. We hope to remove the barrier to entry for data scientists in EO, by reducing the amount of time spent on reformatting datasets. These EO datasets are frequently characterised by very large image formats, high bit-depths, non-standard label formats, pixel values in Digital Number, varied naming conventions, and other dataset-specific peculiarities which slow down development of AI applications.

This package aims to provide users with a pre-prepared dataset ready immediately for AI / Deep Learning applications. The processed datasets are all:
- *Normalised* to reflectance values
- *Resampled* to the same resolution
- *Split* into smaller images for quicker read times
- *Transformed* into one-hot encoded masks
- *Organised* into simple directory tree structure
- *Documented* with useful metadata and command for replication

## Cloud Masking datasets

### **Landsat 8: Biome**<sup>[link](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data)</sup>  *(USGS, 2016)*
96 manually annotated Landsat 8 scenes (~8k-by-8k pixels) from 8 different terrain types (biomes). Data provided at 30m res. for all bands.

### **Landsat 8: SPARCS**<sup>[link](https://www.usgs.gov/land-resources/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation)</sup> *(USGS, 2016)*
80 manually annotated cropped Landsat 8 scenes (1k-by-1k pixels). Data provided at 30m resolution but does not include sharper 'Panchromatic' band.

### **Landsat 7: Irish**<sup>[link](https://landsat.usgs.gov/landsat-7-cloud-cover-assessment-validation-data)</sup> *(USGS, 2016)*
206 manually annotated Landsat 7 scenes from a diverse range of latitudes. Data provided at nominal Landsat 7 resolution of 30m.

### **Sentinel-2: ALCD**<sup>[link](https://zenodo.org/record/1460961#.XYCTRzYzaHt)</sup> *(Baetens et al., 2018)*
38 Sentinel-2 scenes annotated through an "active learning" system. Data provided in native band resolutions (10m - 60m). Does not include the parent scenes, only the masks. Therefore we include a download tool to retrieve the relevant scenes from the Copernicus Open Access Hub, for which a username and password is needed.



## Credits and Contributions

Please use these tools freely in your work. Give this repository an acknowledgement and **always credit and cite the datasets' creators**, who have put a huge amount of work into these labelled datasets!

If you have a dataset that you think would be a good fit, or would like to contribute to the repository, please post an issue, send a PR, or just get in touch!
