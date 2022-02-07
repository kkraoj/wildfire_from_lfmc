# How to calculate plant-water sensitivity?

1. Run `export_lfmc_dfmc_maps.py` to export images to your google drive. Each image is a 15-day aggregate. All images available until today will be exported.
Each image has two bands: first band is LFMC, and second band is DFMC.
1. You can monitor export status at https://code.earthengine.google.com/tasks
1. Once all images are downloaded (will typically take 4 hours), go to your google drive folder. Right click and download. This will download a zipped version of all the maps.
1. Extract the zipped version to any folder of your choice. 
1. Change `data_dir` in `pws_calculation.py` to point to the parent directory where the directory with all maps exists.
1. Run `pws_calculation.py`

