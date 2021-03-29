# Plant-water sensitivity regulates wildfire vulnerability

This repository contains GeoTiff map of plant-water sentivity and scripts required to replicate figures from my manuscript

## Maps

The [maps](maps) folder contains the following maps:
1. Plant-water sensitivity representative of the period 2016 - 2020
1. Vapor pressure deficit trend in hPa/year from PRISM for the period 1980-2020
1. Wildland-urban interface for 1990 and 2010
1. Population density in 1990 and 2010

All maps are for the western US at 4 km resolution.

## Scripts:
The repository consists of scripts in the [analysis](analysis) folder to reproduce figures from the main manuscript. Before running any script, please ensure all [maps](maps) are downloaded. 

1. Run pixel_level_plant_climate_fire.py to reproduce the relationship between plant-water sensitivity and wildfire vulnerability (Fig. 1)
1. Run plant_climate_vs_vpd_trend_absolute.py to reproduce the relationship between plant-water sensitivity and trend in vapor pressure deficit (Fig. 2)
1. Run pop+wui_simplified_landsat_3cats.py to reproduce the growth in WUI population in different plant-water sensitivity zones (Fig. 3)

Rest of the scripts are not needed to reproduce manuscript results. They were used for development of the model and preliminary investigation only.

## Reproducibility guide

1. Clone the repository using `git clone https://github.com/kkraoj/wildfire_from_lfmc.git`
1. Change the directory addresses of `dir_data` and `dir_codes` in `dirs.py`
1. Run any script from the [analysis](analysis) folder to reproduce the figures you wish

## License
Data and scripts presented here are subject to change. Please do not use it until my manuscript's review is completed. Once the manuscript is approved, this section will be changed to reflect the lincense status.

## Issues?

Check the `Issues` tab for troubleshooting or create a new issue.
