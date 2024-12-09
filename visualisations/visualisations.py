import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import os
from haaq_preprocessing import convert_to_weatherbench_grid, format_date
import pickle as pkl

import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# df_gridded = pd.read_csv('haaq_era5.csv')
if os.path.exists('haaq_coords.csv'):
	df = pd.read_csv('haaq_coords.csv')
else:
	df = pd.read_csv('haaq_processed.csv')
	df = df.progress_apply(convert_to_weatherbench_grid, axis=1)
	df.to_csv('haaq_coords.csv')
import sys
pollutant = sys.argv[1] #'RSPM/PM10'
var = df.groupby(['Sampling Date', 'longitude_era5', 'latitude_era5']).count().reset_index()
# var = df.groupby(['Sampling Date', 'longitude_era5', 'latitude_era5']).mean(numeric_only=True).reset_index()
var = var.groupby(['longitude_era5', 'latitude_era5']).mean(numeric_only=True).reset_index()
# var.loc[var[pollutant] < 1, pollutant] = float('NaN')
# nan_count = df.groupby(['longitude_era5', 'latitude_era5']).apply(lambda x: x.isna().sum()).reset_index()
world = gpd.read_file('ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
india = world[world.NAME == "India"]

# Create a plot with geopandas
fig, ax = plt.subplots(figsize=(10, 10))

# Plot India map with only the border and states
india.boundary.plot(ax=ax, color='black', linewidth=1)

# Scatter plot of NO2 values on the map of India
sns.scatterplot(data=var, x='longitude_era5', y='latitude_era5', hue=f'{pollutant}', palette='viridis', size=f'{pollutant}', sizes=(20, 200), ax=ax)

# Add title and labels
plt.title(f'Average # {pollutant} station readings per day')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Save the plot as a high-resolution image
plt.savefig(f'{pollutant.split("/")[0]}_india_grid_loc.png', dpi=600)

# # Show the plot
# plt.show()

# df = df.groupby(['longitude_era5', 'latitude_era5']).mean(numeric_only=True).reset_index()
# for time in df_gridded['Sampling Date']:
# 	breakpoint()
# 	data = df_gridded[df_gridded['Sampling Date'] == time]
# 	for i in range(len(data[0:100])):
# 		latitude, longitude = df_gridded.iloc[i]['latitude_era5'], df_gridded.iloc[i]['longitude_era5']
# 		sub_df = df[(df['longitude_era5'] == longitude) & (df['latitude_era5'] == latitude) & (time == df['Sampling Date'])]
# 		data_loss = sub_df['NO2'].std()
# 		print(data_loss, len(sub_df))
# 		# for i in range(len(values['collated_value'])):
# 		#     ax.plot(lon, lat, marker='o', color='red', markersize=5, transform=ccrs.PlateCarree())  # Plotting the point
# 		#     ax.text(lon + 1, lat + 1, data_loss, fontsize=12, color='blue', transform=ccrs.PlateCarree())  # Adding the label

# # Show the plot
# # plt.show()
