from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy.ma as ma
import pandas as pd
from pylab import *
import numpy as np
import utils
import xarray as xr

'''
--- plot_regional_climate.py

This program calculates the Temperature and precipitation anomalies over CONUS
during 2015-2020 relative to a historical period (1984-2014).

Two plots are output from this program:
  ../Figures/Map_regional_Climate.png
     - Maps of Temperature and Precip anomalies
  ../Figures/T2M_vs_Precip_anom.png
     - Scatter plot of regional temperature vs prcip anomalies
'''

def calculate_annual_mean_data(partial_path, variable, time_dim, lat_range, lon_range):
    '''
    Read & calculate the annual mean of variable and return as xarray.
    Only return values within specified latitude and longitude ranges.
    '''
    #
    annual_mean_list = []
    lat_values = None
    lon_values = None
    #
    for year in range(1980, 2021):
        data = xr.open_dataset(partial_path + str(year).zfill(4) + '.nc')
        # Check if latitude coordinates are in descending order
        if data['lat'][0] > data['lat'][-1]:
            data = data.sel(lat=slice(*lat_range[::-1]), lon=slice(*lon_range))
        else:
            data = data.sel(lat=slice(*lat_range), lon=slice(*lon_range))
        annual_mean_list.append(data[variable].mean(dim=time_dim).assign_coords(year=year))
        if lat_values is None:
            lat_values = data['lat'].values
            lon_values = data['lon'].values
    #
    annual_mean_dataset = xr.concat(annual_mean_list, dim='year')
    #
    return annual_mean_dataset, lat_values, lon_values
    # -----------------------------------------------------------

def calc_anomaly(data,years_clim,years_anom):
    '''
    Calculate anomaly in dataset from anomaly years relative to climatology
    '''
    data_clim = data.sel(year=slice(years_clim[0],years_clim[1])).mean(dim='year')
    data_anom = data.sel(year=slice(years_anom[0],years_anom[1])).mean(dim='year')
    data_diff = data_anom.values - data_clim.values
    return data_diff
    # -----------------------------------------------------------

def plot_maps(lon_precip,lat_precip,precip_anom,lon_T2M,lat_T2M,T2M_anom):
    
    # ========== Start Making Plot =========
    fig = plt.figure(101, figsize=(7.75*0.78,8.5*0.78), dpi=300)
    # --- Precip ---
    # colormap
    cmap1 = plt.cm.RdYlBu
    bounds1= np.arange(16)*0.2-1.5
    norm1 = mpl.colors.BoundaryNorm(bounds1, cmap1.N)
    # Create basemap projection
    m = Basemap(width=4800000,height=3050000,resolution='l',projection='laea',lat_ts=39,lat_0=39.,lon_0=(-155-40)/2.)
    X,Y = np.meshgrid(lon_precip,lat_precip)
    xx,yy=m(X,Y)
    #Create figure
    ax1 = fig.add_axes([0.005,0.045/2.,0.97,0.9/2.])
    tt = m.pcolormesh(xx,yy,ma.masked_invalid(precip_anom),linewidth=0,rasterized=True,cmap=cmap1,norm=norm1)
    m.drawcoastlines(color='k',linewidth=1.0)
    m.drawcountries(color='k',linewidth=0.5)
    m.drawstates(color='k',linewidth=0.5)
    plt.text(0.01, 0.98, '(b)', transform=ax1.transAxes, va='top', ha='left', fontsize=12)
    cbar = plt.colorbar(tt, ax=ax1, ticks=[-1.5,-1,-0.5,0,0.5,1.0,1.5],label='$\mathrm{ \Delta Precip }$ (mm)')
    cbar.ax.xaxis.label.set_rotation(90)  # Rotate the label by 270 degrees
    cbar.ax.xaxis.labelpad = 15  # Set the padding for the label
    # --- T2M ---
    # colormap
    cmap1 = plt.cm.RdYlBu_r
    bounds1= np.arange(16)*0.2-1.5
    norm1 = mpl.colors.BoundaryNorm(bounds1, cmap1.N)
    # Create basemap projection
    m = Basemap(width=4800000,height=3050000,resolution='l',projection='laea',lat_ts=39,lat_0=39.,lon_0=(-155-40)/2.)
    X,Y = np.meshgrid(lon_T2M,lat_T2M)
    xx,yy=m(X,Y)
    # Create figure
    ax1 = fig.add_axes([0.005,1.02/2.,0.97,0.9/2.])
    tt = m.pcolormesh(xx,yy,ma.masked_invalid(T2M_anom),linewidth=0,rasterized=True,cmap=cmap1,norm=norm1)
    m.drawcoastlines(color='k',linewidth=1.0)
    m.drawcountries(color='k',linewidth=0.5)
    m.drawstates(color='k',linewidth=0.5)
    plt.text(0.01, 0.98, '(a)', transform=ax1.transAxes, va='top', ha='left', fontsize=12)
    cbar = plt.colorbar(tt, ax=ax1, ticks=[-1.5,-1,-0.5,0,0.5,1.0,1.5], label='$\mathrm{ \Delta T_{2m} }$ ($^\circ$C)')
    cbar.ax.xaxis.label.set_rotation(90)  # Rotate the label by 270 degrees
    cbar.ax.xaxis.labelpad = 15  # Set the padding for the label
    plt.savefig('../Figures/Map_regional_Climate.png', dpi=300)
    # -----------------------------------------------------------

def calc_regional_Temperature(Regions_05x0625, area_05x0625, T2M_anom):

    Regions_05x0625_union = Regions_05x0625[:,:,0:197-4]
    area_05x0625_union = area_05x0625[:,0:197-4]
    T2M_anom_union = T2M_anom[27:152,4:197]
    T2M_anom_region = np.zeros(7)
    for i in range(7):
        T2M_anom_region[i] = np.sum(T2M_anom_union * Regions_05x0625_union[i,:,:] * area_05x0625_union) / np.sum(Regions_05x0625_union[i,:,:] * area_05x0625_union)
    return T2M_anom_region
    # -----------------------------------------------------------

def calculate_regional_precipitation(lat_005,lon_005,Regions_005,area_005,lat_precip,lon_precip,precip_anom):

    precip_005 = np.zeros((lat_005.size,lon_005.size))
    for i, lat_temp in enumerate(lat_005):
        lat_index = np.abs(lat_precip - lat_temp).argmin()
        for j, lon_temp in enumerate(lon_005):
            lon_index = np.abs( (lon_precip-360) - lon_temp).argmin()
            precip_005[i,j] = precip_anom[lat_index,lon_index]
    #        
    precip_anom_region = np.zeros(7)
    for i in range(7):
        II = np.where(Regions_005==i+1)
        precip_anom_region[i] = np.nansum(precip_005[II] * Regions_005[II] * area_005[II]) / np.nansum(Regions_005[II] * area_005[II])
    return precip_anom_region
    # -----------------------------------------------------------



if __name__ == '__main__':

    # Read data and calculate annual maps
    lat_range = (0, 80)
    lon_range = (-170+360, -40+360)
    annual_mean_precip_dataset, lat_precip, lon_precip = calculate_annual_mean_data('/u/bbyrne1/CPCprecip/precip.','precip','time', lat_range, lon_range)
    # --
    lat_range = (0, 80)
    lon_range = (-170, -40)
    annual_mean_T2M_dataset, lat_T2M, lon_T2M = calculate_annual_mean_data('/u/bbyrne1/MERRA2_daily_mean_T2M','T2M','day', lat_range, lon_range)
    # -----------------------------------
    
    # Calculate 2015-2020 anomalies relative to 1984-2014
    years_climatology = [1984,2014]
    years_anomaly = [2015,2020]
    precip_anom = calc_anomaly(annual_mean_precip_dataset,years_climatology,years_anomaly)
    T2M_anom = calc_anomaly(annual_mean_T2M_dataset,years_climatology,years_anomaly)
    # --------------------------------------------------

    # plot maps
    plot_maps(lon_precip,lat_precip,precip_anom,lon_T2M,lat_T2M,T2M_anom)
    

    ######## Calculate and Plot Regional Anomalies

    # Read / Define arrays
    lat_05x0625, lon_05x0625, Regions_05x0625, area_05x0625 = utils.Regional_mask('05x0625')
    lat_005, lon_005, Regions_005, area_005 = utils.Regional_mask('005')
    Region_Name = ['NW','NGP','MW','SW','SGP','SE','NE']

    # Regional Temperature
    T2M_anom_region = calc_regional_Temperature(Regions_05x0625, area_05x0625, T2M_anom)

    # Regional precip
    precip_anom_region = calculate_regional_precipitation(lat_005,lon_005,Regions_005,area_005,lat_precip,lon_precip,precip_anom)

    # Make scatter plot
    fig = plt.figure(2, figsize=(6.*0.75,4.8*0.75), dpi=300)
    ax1 = fig.add_axes([0.15,0.15,0.83,0.83])
    plt.plot([-0.1,0.425],[0,0],'k:')
    plt.plot([0,0],[-0.1,0.925],'k:')
    for i in range(7):
        plt.text(precip_anom_region[i],T2M_anom_region[i],Region_Name[i])
    plt.xlim([-0.1,0.425])
    plt.ylim([-0.1,0.925])
    plt.xlabel('$\mathrm{\Delta Precip}$ (mm)')
    plt.ylabel('$\mathrm{\Delta T_{2m}}$ (mm)')
    plt.savefig('../Figures/T2M_vs_Precip_anom.png')
