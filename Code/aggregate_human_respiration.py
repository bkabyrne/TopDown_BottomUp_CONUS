import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from math import pi, cos, radians
import utils

def calculate_area(ds, lon_size, lat_size):
    lat = ds['lat'].values
    lon = ds['lon'].values
    earth_radius = 6371009  # in meters
    lat_dist0 = pi * earth_radius / 180.0
    y = lon * 0. + lat_size * lat_dist0
    x = lat * 0.
    for i in range(np.size(lat)):
        x[i] = lon_size * lat_dist0 * cos(radians(lat[i]))
    area = np.zeros((np.size(lat), np.size(lon)))
    for i in range(np.size(lat)):
        for j in range(np.size(lon)):
            area[i, j] = np.abs(x[i] * y[j])

    ds['area'] = (('lat', 'lon'), area)
    return ds

def plot_flux(data, filename='resp.png'):
    # Extracting longitude and latitude
    lon = data['lon'].values
    lat = data['lat'].values
    flux = data['Flux'].values
    # Create a figure
    plt.figure(figsize=(8, 6))
    # Create a Basemap instance
    m = Basemap(projection='merc',
                llcrnrlat=lat.min(), urcrnrlat=lat.max(),
                llcrnrlon=lon.min(), urcrnrlon=lon.max(),
                resolution='i')
    # Create a meshgrid for the longitude and latitude
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y = m(lon_grid, lat_grid)
    # Plot the Flux data
    c_scheme = m.pcolormesh(x, y, flux, shading='auto', cmap='viridis', vmin=0, vmax=10)
    # Add a colorbar
    cb = m.colorbar(c_scheme, location='right', pad='10%')
    cb.set_label('Flux')
    # Add coastlines, states, and countries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    # Add title
    plt.title('Flux Distribution')
    # Save the plot to a file
    plt.savefig(filename)
    plt.close()

# Aligning
nc_human = '../Data_input/human_emissions_density_v3.nc'
human = xr.open_dataset(nc_human)
# Rename the data variable in the Dataset
human = human.rename({'__xarray_dataarray_variable__': 'Flux'})
human = human.rename({'x': 'lon'})
human = human.rename({'y': 'lat'})
## Select the years 2015-2020
#years_to_average = range(2015, 2021)  # 2021 is exclusive
#human_selected_years = human.sel(year=years_to_average)
## Compute the mean over the selected years
#human_mean = human_selected_years.mean(dim='year')
human = human.sortby('lat')
human['Flux'] = human['Flux'].fillna(0)
human = calculate_area(human, 0.1, 0.1)

# Assuming human is already loaded as an xarray Dataset
CONUS_human = human.sel(
    lon=human['lon'].values[540:1150],
    lat=human['lat'].values[1140:1400]
)
plot_flux(CONUS_human.sel(year=2015))

global_total = np.sum(human['Flux'].values * human['area'].values) * 1e-12
regional_total = np.sum(CONUS_human['Flux'].values * CONUS_human['area'].values) * 1e-12

print(f'Global total: {global_total}')
print(f'Regional total: {regional_total}')

# Check for any discrepancies in area and flux calculations
print(f"Total area (global): {np.sum(human['area'].values)}")
print(f"Total area (regional): {np.sum(CONUS_human['area'].values)}")

# Load regional mask and resolution data
lat_res, lon_res, USA_Regions_res, area_res = utils.Regional_mask('005')
lon_res_subset = lon_res[1::2]
lat_res_subset = lat_res[1::2]
USA_Regions_res_subset = USA_Regions_res[1::2, 1::2]
area_res_subset = area_res[1::2, 1::2]
# Create an xarray Dataset
ds_subset = xr.Dataset(
    {
        "Flux": (("lat", "lon"), USA_Regions_res_subset),
        "area_res": (("lat", "lon"), area_res_subset),
    },
    coords={
        "lon": lon_res_subset,
        "lat": lat_res_subset,
    }
)

# Check subsetting
if np.sum(np.abs(lon_res_subset - CONUS_human['lon'].values)) > 0.01 or np.sum(np.abs(lat_res_subset - CONUS_human['lat'].values)) > 0.01:
    raise ValueError("Error in subsetting")

Region_mean = np.zeros((5,7))
Total_area = np.zeros((5,7))
years = np.arange(5)+2015
for yeari, year in enumerate(years):
    print(year)
    CONUS_human_selected_years = CONUS_human.sel(year=year)
    for i in range(7):
        # Find region
        II = np.where(USA_Regions_res_subset == i + 1)
        # Area-weighted mean (gC m-2 yr)
        Region_mean_perArea = np.nansum(CONUS_human_selected_years['Flux'].values[II] * area_res_subset[II]) / np.nansum(area_res_subset[II])
        # Total area
        IIall = np.where(USA_Regions_res == i + 1)
        # Total flux over area (TgC yr)
        Region_mean[yeari,i] = Region_mean_perArea * np.sum(area_res[IIall]) * 1e-12
        Total_area[yeari,i] = np.sum(area_res[IIall])
    

# Create xarray and save                                                                                      
da_a = xr.DataArray(Region_mean, dims=('year','region'), attrs={'units': 'TgC/yr'})
reg_dataset = xr.Dataset({'Human Respiration': da_a})
region_coords = ['Northwest','N Great Plains','Midwest','Southwest','S Great Plains','Southeast','Northeast']
reg_dataset['region'] = region_coords
reg_dataset['year'] = years

reg_dataset.to_netcdf('../Data_processed/Human_respiration_regional.nc')


# Verify regional total sum consistency
print(f"Sum of regional means: {np.sum(Region_mean)}")
print(f"Sum of regional area: {np.sum(Total_area)}")
