import pandas as pd
import os
import sys
sys.path.append('../')
from haaq_preprocessing import convert_to_weatherbench_grid, format_date
import pickle as pkl

import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from matplotlib import rcParams

font_size = 20
rcParams['font.size'] = font_size
rcParams['axes.titlesize'] = font_size
rcParams['axes.labelsize'] = font_size
rcParams['xtick.labelsize'] = font_size
rcParams['ytick.labelsize'] = font_size
rcParams['legend.fontsize'] = font_size

if os.path.exists('../data_files/haaq_coords.csv'):
	df = pd.read_csv('../data_files/haaq_coords.csv')
else:
	df = pd.read_csv('../data_files/haaq_processed.csv')
	df = df.progress_apply(convert_to_weatherbench_grid, axis=1)
	df.to_csv('../data_files/haaq_coords.csv')

df['Sampling Date'] = df['Sampling Date'].apply(format_date)
# df['Sampling Date'] = pd.to_datetime(df['Sampling Date'], format='%d-%m-%Y')

pollutant = sys.argv[1]
choice = int(sys.argv[2])

if choice == 1: # Average number of readings per day
	var = df.groupby(['Sampling Date', 'longitude_era5', 'latitude_era5']).count().reset_index()
	var = var.groupby(['longitude_era5', 'latitude_era5']).mean(numeric_only=True).reset_index()
	title = f'Average # {pollutant} station readings per day'
	save_file = f'{pollutant.split("/")[0]}_india_grid_loc.png'
elif choice == 2: # Average deviation from station vals
	var = df.groupby(['Sampling Date', 'longitude_era5', 'latitude_era5']).std(numeric_only=True).reset_index()
	var = df.groupby(['longitude_era5', 'latitude_era5']).mean(numeric_only=True).reset_index()
	title = f'Average Deviation of {pollutant} from station-level reading'
	save_file = f'{pollutant.split("/")[0]}_india_std.png'


world = gpd.read_file('../ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
india = world[world.NAME == "India"]

fig, ax = plt.subplots(figsize=(10, 10))
india.boundary.plot(ax=ax, color='black', linewidth=1)
sns.scatterplot(data=var, x='longitude_era5', y='latitude_era5', hue=f'{pollutant}', palette='viridis', size=f'{pollutant}', sizes=(20, 200), ax=ax)

plt.title(title)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
legend = ax.legend(loc='lower right')

plt.savefig(save_file, dpi=600)
