import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="Visualise time series data")
parser.add_argument(
    "--variable",
    type=str,
    help="The variable to visualize",
    default="NO2",
)

# Load the CSV file
df = pd.read_csv("../data_files/haaq_physical_vars.csv", index_col=0)

# drop index and "Unnamed: 0" columns
df.drop(columns=["Unnamed: 0"], inplace=True)
df.reset_index(drop=True, inplace=True)

# group by time
df = df.groupby("time").mean()

variable = "NO2"

# plot time vs NO2
plt.plot(df.index, df[variable])
plt.xlabel("Time")
plt.ylabel(f"Average {variable} concentration")
# use sparse x ticks
plt.xticks(df.index[::200], rotation=45)

# add moving average
df["MA"] = df[variable].rolling(window=100).mean()
plt.plot(df.index, df["MA"], label="Moving Average", color="red")

# add title
plt.title(f"{variable} Time Series")

plt.legend()

# Use tight layout
plt.tight_layout()  # Adjusts the plot to ensure everything fits without clipping


# save plot
plt.savefig(f"../plots/{variable}_time_series.png")

# reset plt
plt.clf()


# correlation matrix
corr = df.corr()
# filter out the variables we want to compare
corr = corr[["NO2", "SO2", "RSPM/PM10", "SPM"]]
corr = corr.loc[
    [
        "temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "relative_humidity",
        "specific_humidity",
        "total_precipitation",
        "total_cloud_cover",
    ]
]

sns.heatmap(corr, annot=True)

plt.title("Correlation between Physical & Chemical Variables")

plt.tight_layout()

# save plot
plt.savefig("../plots/correlation_matrix.png")
