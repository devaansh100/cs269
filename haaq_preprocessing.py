import pandas as pd
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import os
import pickle as pkl
import re
import numpy as np

geolocator = Nominatim(user_agent='new-try')
geocode = RateLimiter(geolocator.geocode, max_retries=1, error_wait_seconds=1.)
tqdm.pandas()

def date_parser(row):
	date = row['Sampling Date']
	test = date.split('/')
	date = test if len(test) == 3 else date.split('-')
	if len(date) != 3:
		row['keep'] = False
	else:
		row['Sampling Date'] = '-'.join(date)
	return row

def format_date(date):
  split_date = date.split('-')
  if len(split_date[-1]) == 2:
    prefix = '20' if int(split_date[-1]) < 30 else '19'
    split_date[-1] = prefix + split_date[-1]
  if split_date[1] in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months_to_numbers = {month: str(i) for i, month in enumerate(months)}
    split_date[1] = months_to_numbers[split_date[1]]
  if len(split_date[1]) != 2:
    split_date[1] = '0' + split_date[1]
  if len(split_date[0]) != 2:
    split_date[0] = '0' + split_date[0]
  return '-'.join(split_date)

def string_to_float(s):

    if isinstance(s, str):
	    s = re.sub(r'[^0-9.-]', '', s)  # Remove anything that's not a number or a decimal point or minus
	    s = s.replace('..','.') # Super specific but need it for a few sequences
    return s

def query_loc(query):
	loc = geocode(query)
	time.sleep(1)
	return loc

def convert_to_weatherbench_grid(row, grid_resolution=5.625):
	lats = np.arange(-90 + grid_resolution/2, 90, grid_resolution)
	lon = np.arange(0, 360, grid_resolution)
	snapped_lat = lats[np.abs(lats - row['Latitude']).argmin()]
	snapped_lon = lon[np.abs(lon - row['Longitude']).argmin()]

	row['latitude_era5'], row['longitude_era5'] = snapped_lat, snapped_lon

	return row

def add_manual_keys(cache):
	missed_keys = ['Karimnagar, Andhra Pradesh', 'Nalgonda, Andhra Pradesh', 'Patancheru, Andhra Pradesh', 'Ramagundam, Andhra Pradesh', 'Sangareddy, Andhra Pradesh', 'Warangal, Andhra Pradesh', 'Daranga, Assam', 'Silcher, Assam', 'Dhanbad, Bihar', 'Jharia, Bihar', 'Sindri, Bihar', 'Assanora, Goa', 'Tilamol, Goa', 'Anklesvar, Gujarat', 'Turicorin, Tamil Nadu', 'Rai Bareilly, Uttar Pradesh', 'Delhi, Delhi']
	correct_keys = ['Karimnagar', 'Nalgonda', 'Patancheru', 'Ramagundam', 'Sangareddy', 'Warangal', 'Baksa District, Assam', 'Silchar, Assam', 'Dhanbad, Jharkhand', 'Jharia, Jharkhand', 'Sindri, Jharkhand', 'Assonora, Goa', 'Tilamola, Goa', 'Ankleshwar, Gujarat', 'Tuticorin, Tamil Nadu', 'Raebareli, Uttar Pradesh', 'Delhi, India']
	assert len(missed_keys) == len(correct_keys)
	for original_key, fixed_key in tqdm(zip(missed_keys, correct_keys), total=len(missed_keys)):
		cache[original_key] = geocode(fixed_key)
		time.sleep(1)
	return cache

def main():
	if not os.path.exists('haaq_processed.csv'):
		csv_files = glob.glob('data_files/cpcb*csv')
		data = []
		for filename in tqdm(csv_files):
			try:
					df = pd.read_csv(filename, encoding='ISO-8859-1')
					if 'PM 2.5' in df.columns:
						df = df.rename(columns={'PM 2.5': 'SPM'})
					data.append(df)
			except Exception as e:
				print(f'{filename}: {e}')

		df = pd.concat(data, axis=0, ignore_index=True)
		df['keep'] = True

		# Remove rows which are not daily assimilations
		df = df.progress_apply(date_parser, axis=1)
		df = df[df['keep']]
		df.drop(columns=['keep', 'Stn Code'], inplace=True)
		df = df.sort_values(by = 'Sampling Date')

		df['NO2'] = df['NO2'].progress_apply(string_to_float)
		df['NO2'] = df['NO2'].astype(float)
		df['SPM'] = df['SPM'].progress_apply(string_to_float)
		df['SPM'] = df['SPM'].astype(float)
		df = df.groupby(['Sampling Date', 'State', 'City/Town/Village/Area'], as_index=False).mean(numeric_only=True).reset_index()
		df['Longitude'], df['Latitude'] = float('Nan'), float('Nan')
		if not os.path.exists('cache.pkl'):
			
			keys = set([f"{df.iloc[i]['City/Town/Village/Area']}, {df.iloc[i]['State']}" for i in range(len(df))])
			cache = {}
			
			for key in tqdm(keys, desc="Getting coordinates"):
				cache[key] = query_loc(key)

			cache = add_manual_keys(cache)
			with open('cache.pkl', 'wb') as f:
				pkl.dump(cache, f)
		else:
			with open('cache.pkl', 'rb') as f:
				cache = pkl.load(f)
		
		for i in range(len(df)):
			key = f"{df.iloc[i]['City/Town/Village/Area']}, {df.iloc[i]['State']}"
			if cache[key] is not None:
				df.loc[i, 'Latitude'] = cache[key].latitude
				df.loc[i, 'Longitude'] = cache[key].longitude
		df.to_csv('haaq_processed.csv')
	else:
		df = pd.read_csv('haaq_processed.csv')

	df['longitude_era5'], df['latitude_era5'] = None, None
	df = df.progress_apply(convert_to_weatherbench_grid, axis=1)
	df['Sampling Date'] = df['Sampling Date'].apply(format_date)
	df = df.groupby(['Sampling Date', 'longitude_era5', 'latitude_era5']).mean(numeric_only=True).reset_index()
	df.drop(columns=['Longitude', 'Latitude', 'Unnamed: 0'], inplace=True)
	df = df[df['Sampling Date'] != '31-11-2014']
	df = df[df['Sampling Date'] != '31-06-2014']
	df = df[df['Sampling Date'] != '31-09-2014']
	df.to_csv('haaq_era5_5625.csv')


if __name__ == '__main__':
	main()
