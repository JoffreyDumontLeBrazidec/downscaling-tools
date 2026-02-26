import cartopy.crs as crs  # can't be called after metview for some reason
import metview as mv


# load modules


import matplotlib.pyplot as plt
import xarray as xr


import numpy as np
from time import sleep
import os as os
import glob as glob
import netCDF4 as nc
import seaborn as sns
from icecream import ic
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*decode_timedelta will default to False.*",
    category=FutureWarning,
    module="cfgrib.xarray_plugin",
)


sns.set_theme(style="ticks", rc={"font.family": "DejaVu Sans"})


# Setting some script parameters
ensemble_size = 10

# select subdomain and resolution
lat = np.arange(11, 39.01, 0.25)
lon = np.arange(-99, -71.01, 0.25)
lon2, lat2 = np.meshgrid(lon, lat)
lat_up = np.max(lat)
lat_down = np.min(lat)
long_down = np.min(lon)
long_up = np.max(lon)
resol = 0.25
year = "2023"
month = "08"

dates = [28]
time = 1  # i.e. 96h
members = [1, 2, 5, 7, 9]

# Define experiment IDs
list_expid_ml = ["ENFO_O320_ip6y", "ENFO_O320_ioj2"]
expid_enfo_O320 = "ENFO_O320_0001"
expid_eefo_O96 = "EEFO_O96_0001"
analysis = "OPER_O320_0001"

list_exps = [expid_eefo_O96] + list_expid_ml + [expid_enfo_O320, analysis]

main_dir = "/home/ecm5702/hpcperm/data/tc/idalia"

# retrieve the data
from tools.loading_data import DataRetriever

# Initialize the retriever
retriever = DataRetriever(
    main_dir, dates, year, month, resol, lat_down, lat_up, long_down, long_up
)

# Retrieve all data
msl, wind10m = retriever.retrieve_all_data(
    analysis,
    expid_enfo_O320,
    expid_eefo_O96,
    list_expid_ml,
)


from plots.fields import plot_member_data


# Apply the function for mean sea level pressure
sns.set_theme(style="ticks", rc={"font.family": "DejaVu Sans"})
plot_member_data(
    {key: value for key, value in msl.items() if "OPER" not in key},
    lon2,
    lat2,
    levels=np.linspace(985, 1015, 31),
    title="Mean Sea Level Pressure",
    filename=f"idalia_msl_fields_{dates[0]}_{month}_step{time}.png",
    members=members,
    time=time,
    day=0,
)

# Apply the function for wind speed 10m
plot_member_data(
    {key: value for key, value in wind10m.items() if "OPER" not in key},
    lon2,
    lat2,
    levels=np.linspace(0, 30, 31),
    title="Wind Speed 10m",
    filename=f"idalia_wind10m_fields_{dates[0]}_{month}_step{time}.png",
    members=members,
    time=time,
    day=0,
)
