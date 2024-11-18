# HLMamba
This an official Pytorch implementation of our paper ["Joint Classification of Hyperspectral and LiDAR Data Based on Mamba"](https://ieeexplore.ieee.org/document/10679212). The specific details of the framework are as follows.
![HLMamba](./HLMamba.png)
****
# Datasets
- [The Houston2013 dataset](https://hyperspectral.ee.uh.edu/?page_id=459)
includes a hyperspectral image (HSI) and a LiDAR-based digital surface model (DSM), collected by the National Center for Airborne Laser Mapping (NCALM) using the ITRES CASI-1500 sensor over the University of Houston campus in June 2012. The HSI comprise 144 spectral bands covering a wavelength range from 0.38 to 1.05 $\mu m$ while LiDAR data are provided for a single band. Both the HSI and LiDAR data share dimensions of 349 × 1905 pixels with a spatial resolution of 2.5 $m$. The dataset contains 15 categories, with a total of 15,029 real samples available.
- [The MUUFL dataset](https://github.com/GatorSense/MUUFLGulfport)
was acquired in November 2010 over the area of the campus of University of Southern Mississippi Gulf Park, Long Beach Mississippi, USA. The HSI data was gathered using the ITRES Research Limited (ITRES) Compact Airborne Spectral Imager (CASI-1500) sensor, initially comprising 72 bands. Due to excessive noise, the first and last eight spectral bands were removed, resulting in a total of 64 available spectral channels ranging from 0.38 to 1.05 $\mu m$. LiDAR data was captured by an ALTM sensor, containing two rasters with a wavelength of 1.06 $\mu m$. The dataset consists of 53,687 groundtruth pixels, encompassing 11 different land-cover classes.
