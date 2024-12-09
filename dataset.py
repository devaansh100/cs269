import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import yaml
from types import SimpleNamespace
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import pandas as pd

def dict_to_namespace(data):
	if isinstance(data, dict):
		return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in data.items()})
	elif isinstance(data, list):
		return [dict_to_namespace(item) if isinstance(item, dict) else item for item in data]
	else:
		return data

def nanstd(input, dim=None, keepdim=False, unbiased=True):
	# Replace NaNs with zeros
	input_without_nans = input.clone()
	input_without_nans[input.isnan()] = 0

	# Count the number of non-NaN elements
	count = torch.sum(~input.isnan(), dim=dim, keepdim=keepdim)

	# Compute the mean of non-NaN values
	mean = torch.sum(input_without_nans, dim=dim, keepdim=keepdim) / count

	try:
		mean = mean[:, :, None, None]
	except:
		mean = mean[None, :]

	# Compute the variance of non-NaN values
	variance = torch.sum((input_without_nans - mean) ** 2, dim=dim, keepdim=keepdim) / (count - 1 if unbiased else count)

	# Return the standard deviation
	return torch.sqrt(variance)

class ERA5Pollutants(Dataset):
	def __init__(self, config, mode = 'train', means = None, stds = None):
		super().__init__()
		if config.load_data:
			data = pd.read_csv(config.load_data)
		else:
			data = pd.read_csv(config.haaq.path)
			assert config.processed_data != '', 'Pass processed_data in the config - file will be stored here'
			era5 = xr.open_zarr(config.era5.path)
			daily_avg = era5.coarsen(time=4).mean()
			daily_avg = daily_avg.assign_coords(time=daily_avg["time"].dt.floor("D"))
			del era5

			new_lats = np.arange(-90 + config.haaq.res/2, 90, config.haaq.res)
			new_lons = np.arange(0, 360, config.haaq.res)
			daily_avg_subset = daily_avg.interp(latitude=new_lats, longitude=new_lons, method='linear')
			del daily_avg

			for var in list(daily_avg_subset.data_vars):
			if var not in config.input_vars:
				daily_avg_subset = daily_avg_subset.drop_vars(var)
			else:
				data[var] = 0.

			drop_level = [50, 100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925]
			daily_avg_subset = daily_avg_subset.drop_sel(level=drop_level)
			daily_avg_subset.load()

			for i in tqdm(range(len(data))):
				for var in config.output_vars:
					data[i, var] = daily_avg_subset.sel(time=data.iloc[i]['time'], latitude=data.iloc[i]['latitude'], longitude=data.iloc[i]['longitude'])[0].values().item()

			data.to_csv(config.processed_data)

		
		data['time'] = pd.to_datetime(data['time'])
		if mode == 'train':
			time_mask = (data['time'] >= config.train.time_slice[0]) & (data['time'] <= config.train.time_slice[1])
		elif mode == 'val':
			time_mask = (data['time'] >= config.val.time_slice[0]) & (data['time'] <= config.val.time_slice[1])
		elif mode == 'test':
			time_mask = (data['time'] >= config.test.time_slice[0]) & (data['time'] <= config.test.time_slice[1])
		else:
			raise ValueError('Unknown data mode, use train/val/test')
		data = data[time_mask]
		data[config.output_vars] = data[config.output_vars].interpolate(axis=0)
		data[config.output_vars] = data[config.output_vars].bfill()
		if means is None or stds is None:
			means, stds = {}, {}
			for var in config.input_vars + config.output_vars:
				means[var] = data[var].mean()
				stds[var] = data[var].std()
		for var in config.input_vars + config.output_vars:
			data[var] = (data[var] - means[var]) / stds[var]

		self.means, self.stds = means, stds
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		inputs, outputs = [], []
		for input_var in config.input_vars:
			inputs.append(self.data[input_var].iloc[idx].astype(np.float32))

		for output_var in config.output_vars:
			outputs.append(self.data[output_var].iloc[idx].astype(np.float32))

		return torch.tensor(inputs), torch.tensor(outputs)

	@classmethod
	def collate_fn(cls, batch):
		x_batch, y_batch = zip(*batch)
		return torch.stack(x_batch), torch.stack(y_batch)

if __name__ == '__main__':
	config = dict_to_namespace(yaml.safe_load(open('config.yaml').read()))
	train_ds = ERA5Pollutants(config, mode = 'train', iterate_over='locations')
	train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=train_ds.collate_fn)
	breakpoint()

	val_dataset = ERA5Pollutants(config, mode = 'val')
	val_dl = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=train_dataset.collate_fn)

	test_dataset = ERA5Pollutants(config, mode = 'test')
	test_dl = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=train_dataset.collate_fn)

