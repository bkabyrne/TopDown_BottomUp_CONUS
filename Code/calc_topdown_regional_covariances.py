# --- import modules ---   
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import utils

'''

calc_topdown_regional_covariances.py

This program reads in the OCO-2 v10 MIP and aggregates the models into regional totals for each MIP
experiment then calculates the correlation matrix between regions.

Output:
 - '../Figures/text_covariance.png'
     Plot of the correlation between regions

'''

def read_MIP_net_flux(models,experiment):

    Net_Flux = np.zeros((len(models),6,180,360))
    
    nc_file = '../Data_processed/MIPv10_'+experiment+'_ens.nc'
    f=Dataset(nc_file,mode='r')
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    year = f.variables['year'][:]
    FF = f.variables['FF'][:]
    for i, model in enumerate(models):
        Net_Flux[i,:,:,:] = (f.variables[model+'_NBE'][:] + f.variables[model+'_ocean'][:] + FF)  * 12./44. # gCO2/m2/yr  # 12gC/44gCO2

    return Net_Flux


def calculate_regional_fluxes(Net_Flux,USA_Regions_1x1,area_1x1):

    # 1 = Northwest
    # 2 = N Great Plains
    # 3 = Midwest
    # 4 = Southwest
    # 5 = S Great Plains
    # 6 = Southeast
    # 7 = Northeast
    # 8 Alaksa
        
    Region_net_all = np.zeros((11,6,7)) # (model,year,region)
    area_all = np.zeros((11,6,7))
    for k in range(6):
        for im in range(11):
            TEMP = Net_Flux[im,k,:,:]
            for j in range(7):
                #
                USA_Regions_1x1_temp = USA_Regions_1x1[j,:,:]
                #
                Region_net_all[im,k,j] = np.sum(TEMP * USA_Regions_1x1_temp * area_1x1) * 1e-12 # TgC/yr
                area_all[im,k,j] = np.sum(USA_Regions_1x1_temp * area_1x1) # m2
    
    # For mean year
    Region_net_all_meanYr = np.mean(Region_net_all,1)

    return Region_net_all_meanYr

# ====

if __name__ == '__main__':
    
    lat, lon, Regions, area = utils.Regional_mask('1x1')
    models = ['Ames', 'Baker', 'CAMS', 'COLA', 'CSU', 'CT', 'NIES', 'OU', 'TM54DVar', 'UT', 'WOMBAT']
    region_coords = ['Northwest','N Great Plains','Midwest','Southwest','S Great Plains','Southeast','Northeast']
    time_coords = [2015,2016,2017,2018,2019,2020]
        
    all_exp_dict_year = {}
    all_exp_dict = {}
    experiments = ['Prior','IS','LNLG','LNLGIS','LNLGOGIS']
    for experiment in experiments:
        Net_Flux =  read_MIP_net_flux(models,experiment)
        Region_meanYr = calculate_regional_fluxes(Net_Flux,Regions,area)
        Region_corrMatrix = np.corrcoef(Region_meanYr.T)
        all_exp_dict[experiment] = (['region', 'region'], Region_corrMatrix)

    MIP_regional_data = xr.Dataset(all_exp_dict)

    region_labels = ['NW', 'NGP', 'MW', 'SW', 'SGP', 'SE', 'NE']

    axvals = [ [0.1/2.,2.1/3.,0.8/2.,0.8/3.],
               [0.1/2.,1.1/3.,0.8/2.,0.8/3.],
               [1.1/2.,1.1/3.,0.8/2.,0.8/3.],
               [0.1/2.,0.1/3.,0.8/2.,0.8/3.],
               [1.1/2.,0.1/3.,0.8/2.,0.8/3.] ]
        
    fig = plt.figure(figsize=(8.5*0.875, 10.*0.875))
    for i, exp in enumerate(experiments):
        ax1 = fig.add_axes(axvals[i])
        plt.imshow(MIP_regional_data[exp].values, vmin=-1, vmax=1, cmap='RdYlBu_r', interpolation='nearest')
        plt.xticks(np.arange(len(region_labels)), region_labels)
        plt.yticks(np.arange(len(region_labels)), region_labels)
        plt.colorbar()
        plt.title(exp)
    plt.savefig('../Figures/text_covariance.png')
