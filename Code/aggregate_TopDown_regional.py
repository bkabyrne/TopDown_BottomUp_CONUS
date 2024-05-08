# --- import modules ---   
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import utils

'''

Aggregate_TopDown_regional.py

This program reads in the OCO-2 v10 MIP and aggregates the models into regional totals for each MIP

Output:
 - Regional_OCO2_MIP.nc

'''

def read_MIP_net_flux(models,experiment):

    Net_Flux = np.zeros((len(models),6,180,360))
    
    nc_file = '/u/bbyrne1/python_codes/GST_v10MIP/MIPv10_'+experiment+'_ens.nc'
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
    
    # For year
    Region_net_percentile_75 = np.percentile(Region_net_all,75, axis=0)
    Region_net_percentile_50 = np.percentile(Region_net_all,50, axis=0)
    Region_net_percentile_25 = np.percentile(Region_net_all,25, axis=0)
    Region_net_std = np.abs(Region_net_percentile_75 - Region_net_percentile_25)/1.35
    #
    # For mean year
    Region_net_all_meanYr = np.mean(Region_net_all,1)
    Region_net_meanYr_percentile_75 = np.percentile(Region_net_all_meanYr,75, axis=0)
    Region_net_meanYr_percentile_50 = np.percentile(Region_net_all_meanYr,50, axis=0)
    Region_net_meanYr_percentile_25 = np.percentile(Region_net_all_meanYr,25, axis=0)
    Region_net_meanYr_std = np.abs(Region_net_meanYr_percentile_75 - Region_net_meanYr_percentile_25)/1.35

x    Region_median_std = np.concatenate( ( np.expand_dims(Region_net_percentile_50,axis=0),
                                          np.expand_dims(Region_net_std,axis=0) ),
                                        axis=0)
    
    Region_meanYr_median_std = np.concatenate( ( np.expand_dims(Region_net_meanYr_percentile_50,axis=0),
                                                 np.expand_dims(Region_net_meanYr_std,axis=0) ),
                                               axis=0)

    return Region_meanYr_median_std, Region_median_std

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
        Region_meanYr_median_std, Region_median_std = calculate_regional_fluxes(Net_Flux,Regions,area)
        all_exp_dict_year[experiment] = (['stat', 'year', 'region'], Region_median_std, {'units': 'TgC/yr'})
        all_exp_dict[experiment] = (['stat', 'region'], Region_meanYr_median_std, {'units': 'TgC/yr'})

    MIP_regional_data_year = xr.Dataset(all_exp_dict_year)
    MIP_regional_data_year['region'] = region_coords
    MIP_regional_data_year['year'] = time_coords
    MIP_regional_data_year['stat'] = ['median','std']
    MIP_regional_data_year.to_netcdf('../Data_processed/Regional_OCO2_MIP_year.nc')

    MIP_regional_data = xr.Dataset(all_exp_dict)
    MIP_regional_data['region'] = region_coords
    MIP_regional_data['stat'] = ['median','std']
    MIP_regional_data.to_netcdf('../Data_processed/Regional_OCO2_MIP_2015to2020avg.nc')

