from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy.ma as ma
import pandas as pd
from pylab import *
import numpy as np
import utils

'''
plot_CONUS_budget.py

Creates a bar plot of CONUS totals for LNLGIS and all bottom-up fluxes

output:
  '../Figures/CONUS_TopDown_LNLGIS_vs_bottomup_20240412.png'

'''

def build_bar(ax,width, xvals, rf, fcolors, ymin, ymax, ytickvec, Rlabel):
    #ax = inset_axes(ax, width=width, \
    #                height=width*0.8, \
    #                loc=3, \
    #                bbox_to_anchor=(mapx, mapy), \
    #                bbox_transform=ax.transData, \
    #                borderpad=0, \
    #                axes_kwargs={'alpha': 0.35, 'visible': True})
    #                                                                                              
    xvals_float = np.asarray(xvals, dtype=np.float32)
    plt.plot([np.min(xvals_float-width)-width, np.max(xvals_float+width)+width],[0,0],'k',linewidth=1.)
    plt.plot([(xvals_float[0]-width+xvals_float[1])/2.,(xvals_float[0]-width+xvals_float[1])/2.],[ymin,ymax],'k:',linewidth=1.0)
    plt.fill_between([(xvals_float[1]+xvals_float[2]+width)/2.,np.max(xvals_float+width)+width],[ymin,ymin],[ymax,ymax],color='grey',alpha=0.18)
    #                                                                                              
    yvals = [ rf['LNLGIS median'],
              rf['Bottom-up NCE'],
              rf['FF and IPPU'],
              rf['Bottom-up dC'],
              rf['Crop yield'],
              rf['Forest Harvest'],
              rf['Respiration'],
              rf['Biofuel'],
              rf['Others'] ]
    #    
    yerr = rf['LNLGIS std']
    #
    bar_iter = 0
    for x,y,c in zip(xvals, yvals, fcolors):
        #                                                                                          
        if bar_iter == 0:
            ax.bar(x-width, y, yerr=yerr, width=1.0, fc=c,capsize=10,edgecolor='k',linewidth=1.,hatch='//',alpha=0.5)
            plt.text(np.min(xvals_float-width)-width*0.7,ymin+(ymax-ymin)*0.005,'Top-Down',ha='left',va='bottom',fontsize=12.)
            bar_iter = 1
        elif bar_iter == 1:
            ax.bar(x, y, width=1.0, fc=c,capsize=5,edgecolor='k',linewidth=1.)
            plt.text(x-width*2.75/3.,ymin+(ymax-ymin)*0.005,'Bottom-Up',ha='left',va='bottom',fontsize=12.)
            bar_iter = 2
        else:
            ax.bar(x+width, y, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5)
    plt.arrow(np.max(xvals_float+width)+width*2./3.,50*4.,0,100*4.,width=0.07,head_width=0.4,head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))),fc='k')
    plt.text(np.max(xvals_float+width)+width*1.6/3.,100*4.,'source',ha='right',va='center',fontsize=12.5)
    plt.arrow(np.max(xvals_float+width)+width*2./3.,-50*4.,0,-100*4.,width=0.07,head_width=0.4,head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))),fc='k')
    plt.text(np.max(xvals_float+width)+width*1.6/3.,-100*4.,'sink',ha='right',va='center',fontsize=12.5)
    plt.xlim([np.min(xvals_float-width)-width, np.max(xvals_float+width)+width])
    plt.text(np.min(xvals_float-width)-width + ((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))*0.99,ymin+(ymax-ymin)*0.99,Rlabel,ha='right',va='top',fontsize=14)
    plt.xticks([])#range(len(xvals)), xvals, fontsize=10, rotation=30)                             
    ax.set_xticklabels([])
    plt.ylim([ymin,ymax])
    plt.yticks(ytickvec)
    ax.axis('on')
    plt.tick_params(axis='y', labelsize=14)
    return ax

lat, lon, Regions, area = utils.Regional_mask('005')
Regions[np.where(Regions==0)]=np.nan

Region_label = ['Northwest','N Great Plains','Midwest','Southwest','S Great Plains','Southeast','Northeast']
Regional_CO2_Budget = pd.read_csv('../Data_processed/Regional_CO2_Budget.csv')
Regional_CO2_Budget.set_index('Region', inplace=True)

Regional_CO2_Budget['Bottom-up NCE'] = (Regional_CO2_Budget['Biofuel wood'] + 
                                        Regional_CO2_Budget['Biofuel ethanol'] + 
                                        Regional_CO2_Budget['Biofuel biodiesal'] + 
                                        Regional_CO2_Budget['Incineration'] + 
                                        Regional_CO2_Budget['FF and IPPU'] + 
                                        Regional_CO2_Budget['Crop yield'] + 
                                        Regional_CO2_Budget['Livestock Respiration'] + 
                                        Regional_CO2_Budget['Human Respiration'] + 
                                        Regional_CO2_Budget['Forest Harvest'] + 
                                        Regional_CO2_Budget['Forest inventory'] + 
                                        Regional_CO2_Budget['grassland stockchange'] + 
                                        Regional_CO2_Budget['cropland stockchange'] +
                                        Regional_CO2_Budget['residual wood'] +
                                        Regional_CO2_Budget['PIC and SWDS stockchange'] +
                                        Regional_CO2_Budget['wood trade'] +
                                        Regional_CO2_Budget['residual crop'] +
                                        Regional_CO2_Budget['crop landfill stockchange'] +
                                        Regional_CO2_Budget['crop trade'] +
                                        Regional_CO2_Budget['Lake and River emissions'] +
                                        Regional_CO2_Budget['Lake and River carbon burial'] +
                                        Regional_CO2_Budget['Coastal carbon export'] ) 

Regional_CO2_Budget['Bottom-up dC'] = ( Regional_CO2_Budget['Forest inventory'] + 
                                        Regional_CO2_Budget['grassland stockchange'] + 
                                        Regional_CO2_Budget['cropland stockchange'] +
                                        Regional_CO2_Budget['PIC and SWDS stockchange'] +
                                        Regional_CO2_Budget['crop landfill stockchange'] +
                                        Regional_CO2_Budget['Lake and River carbon burial'] )

Regional_CO2_Budget['Respiration'] = ( Regional_CO2_Budget['Livestock Respiration'] +
                                    Regional_CO2_Budget['Human Respiration'] )

Regional_CO2_Budget['Biofuel'] = ( Regional_CO2_Budget['Biofuel wood'] +
                                   Regional_CO2_Budget['Biofuel ethanol'] +
                                   Regional_CO2_Budget['Biofuel biodiesal'] )

Regional_CO2_Budget['Others'] = ( Regional_CO2_Budget['Incineration'] + 
                                  Regional_CO2_Budget['residual wood'] +
                                  Regional_CO2_Budget['wood trade'] +
                                  Regional_CO2_Budget['residual crop'] +
                                  Regional_CO2_Budget['crop trade'] +
                                  Regional_CO2_Budget['Lake and River emissions'] +
                                  Regional_CO2_Budget['Coastal carbon export'] ) 

CONUS_CO2_Budget = Regional_CO2_Budget.sum(axis=0)



#Create figure
fig = plt.figure(1, figsize=(7.50*0.95,5.05*0.95), dpi=300)
ax1 = fig.add_axes([0.155,0.173,0.83,0.81])


# Create colors
Inv_cmap_test1 = cm.get_cmap('viridis')
Inv_colorst = Inv_cmap_test1(np.linspace(0, 0.9, 5))
Inv_colors = ['grey',Inv_colorst[0],Inv_colorst[1],Inv_colorst[3],Inv_colorst[4],'grey']
colors = [Inv_colors[3],'grey','black','green','orange','saddlebrown','olive','darkslategrey','pink']

# Make inset plots
bax = build_bar(ax1, 1.0, [1,2,3,4,5,6,7,8,9],CONUS_CO2_Budget,colors, -200*4., 370*4., [-400,0,400,800,1200],'CONUS')

# Create lengend
patch0 = mpatches.Patch(color=colors[0], alpha=0.5, hatch='//', label='NCE (top-down)')
patch1 = mpatches.Patch(color=colors[1], label='NCE (bottom-up)')
patch2 = mpatches.Patch(color=colors[2], label='Fossil fuel')
patch3 = mpatches.Patch(color=colors[3], label='C Stockchange')
patch4 = mpatches.Patch(color=colors[4], label='Crop harvests')
patch5 = mpatches.Patch(color=colors[5], label='Wood harvests')
patch6 = mpatches.Patch(color=colors[6], label='Human+Livestock\nrespiration')
patch7 = mpatches.Patch(color=colors[7], label='Biofuels')
patch8 = mpatches.Patch(color=colors[8], label='Others')
ax1.legend(handles=[patch0,patch1,patch2,patch3,patch4,patch5,patch6,patch7,patch8],loc='upper center', bbox_to_anchor=(0.5, 0.017),ncol=3,frameon=False,fontsize=11.5)
plt.ylabel('Carbon flux (TgC year$^{-1}$)',fontsize=14)
# Save figure
plt.savefig('../Figures/CONUS_TopDown_LNLGIS_vs_bottomup_20240412.png', dpi=300)

