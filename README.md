# Air Pollutant Tracking Data
This repository contains the code and data for the paper "Enhancing Air Pollutant Tracking with ERA5 for Underserved Regions". This project was undertaken as part of the course project for CS269: AI For Climate Change, by Prof. Aditya Grover at UCLA in Fall 2024.

Here, we add combine the HAAQ dataset, provided by the CPCB of India with physical variables from ERA5 to improve air pollutant tracking. We aim to create a machine-learning ready dataset for this task.

For dataset statistics, task description, baselines and visualisations, please refer to the paper. Here, we will only describe how to run the code and use the data.

## Downloading the HAAQ data files
The HAAQ files are stored in the CPCB databased in a templatised format. This format can be seen in ```download_data.sh```. To download the files, we execute all possible commands using that template; not all of them will return valid files. While this can be improved, we manage to extra 601/645 files in the database. These have also been provided in ```data_files/data.zip```.
## Processing the HAAQ data files
To process the dataset, run ```pytho haaq_preprocessing.py```. This generates an intermediate file, ```haaq_processed.csv```, and ends with generating ```haaq_era5_5625.csv```. The ```grid_resolution``` can be changed in the code to create higher resolution data. This code also generates a ```cache.pkl```, which contains the coordinates of the towns in the dataset, retrieved via an API. We also provide all these files in ```data_files```.
## Combining with ERA5 for model training
We use a ```config.yaml``` to control the data and model training, where each paramter is self-explanatory. The time slices for train, val and test can be modified depending on the use case. The default parameters correspond to the standardisation of the years, as defined in the paper. This notebook can be executed in Google Colab. Note that ```config.yaml``` and ```data_files/haaq_physical_vars.csv``` will need to be uploaded.

In case you would like to change the ```input_vars``` in the config, a new dataset would need to be generated. In this case, set ```load_data = ''``` and ```processed_data``` to the target file before executing the code. This will trigger downloading the ERA5 subset from Google Cloud, and combining with HAAQ to create the augmented dataset. Note that this process will take time, since the reads from GCS can be quite slow.
