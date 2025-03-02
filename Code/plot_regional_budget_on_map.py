from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy.ma as ma
import pandas as pd
from pylab import *
import numpy as np
import utils

def build_bar(mapx, mapy, ax, width, xvals, rf, fcolors, ymin, ymax, ytickvec, Rlabel):
    ax_h = inset_axes(ax, width=width, \
                    height=width*0.8, \
                    loc=3, \
                    bbox_to_anchor=(mapx, mapy), \
                    bbox_transform=ax.transData, \
                    borderpad=0, \
                    axes_kwargs={'alpha': 0.35, 'visible': True})
    #                                                                                              
    xvals_float = np.asarray(xvals, dtype=np.float32)
    plt.plot([np.min(xvals_float-width)-width, np.max(xvals_float+width)+width],[0,0],'k',linewidth=0.5)
    plt.plot([(xvals_float[0]-width+xvals_float[1])/2.,(xvals_float[0]-width+xvals_float[1])/2.],[ymin,ymax],'k:',linewidth=1.0)
    plt.fill_between([(xvals_float[1]+xvals_float[2]+width)/2.,np.max(xvals_float+width)+width],[ymin,ymin],[ymax,ymax],color='grey',alpha=0.18)
    #                                                                                              
    yvals = [ rf['LNLGIS median'],
              rf['Bottom-up NCE'],
              rf['FF and IPPU'],
              rf['Bottom-up dC'],
              rf['Crop Harvest (adjusted)'],
              rf['Forest Harvest (adjusted)'],
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
            ax_h.bar(x-width, y, yerr=yerr, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5,hatch='////',alpha=0.5)
            plt.text(x-width,ymin+(ymax-ymin)*0.005,'TD',ha='center',va='bottom',fontsize=6.5)
            bar_iter = 1
        elif bar_iter == 1:
            ax_h.bar(x, y, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5)
            plt.text(x-width*2.25/3.,ymin+(ymax-ymin)*0.005,'Bottom-Up',ha='left',va='bottom',fontsize=6.5)
            bar_iter = 2
        else:
            ax_h.bar(x+width, y, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5)
    plt.arrow(np.max(xvals_float+width)+width*2./3.,50,0,100,width=0.07,head_width=0.4,head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))),fc='k')
    plt.text(np.max(xvals_float+width)+width*1.6/3.,100,'source',ha='right',va='center',fontsize=6.5)
    plt.arrow(np.max(xvals_float+width)+width*2./3.,-50,0,-100,width=0.07,head_width=0.4,head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))),fc='k')
    plt.text(np.max(xvals_float+width)+width*1.6/3.,-100,'sink',ha='right',va='center',fontsize=6.5)
    plt.xlim([np.min(xvals_float-width)-width, np.max(xvals_float+width)+width])
    plt.text(np.min(xvals_float-width)-width + ((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))*0.99,ymin+(ymax-ymin)*0.99,Rlabel,ha='right',va='top',fontsize=7)
    plt.xticks([])#range(len(xvals)), xvals, fontsize=10, rotation=30)                             
    ax_h.set_xticklabels([])
    plt.ylim([ymin,ymax])
    plt.yticks(ytickvec)
    ax_h.axis('on')
    return ax_h

lat, lon, Regions, area = utils.Regional_mask('005')
Regions[np.where(Regions==0)]=np.nan

Region_label = ['Northwest','N Great Plains','Midwest','Southwest','S Great Plains','Southeast','Northeast']
Regional_CO2_Budget = pd.read_csv('../Data_processed/Regional_CO2_Budget_Summary.csv')
Regional_CO2_Budget.set_index('Region', inplace=True)

Regional_TopDown_NCE = pd.read_csv('../Data_processed/Regional_TopDown_NCE.csv')
Regional_TopDown_NCE.set_index('region', inplace=True)
Regional_CO2_Budget['LNLGIS median'] = Regional_TopDown_NCE['LNLGIS median'].copy()
Regional_CO2_Budget['LNLGIS std'] = Regional_TopDown_NCE['LNLGIS std'].copy()

#Regional_CO2_Budget['Bottom-up NCE'] = (Regional_CO2_Budget['Biofuel wood'] + 
#                                        Regional_CO2_Budget['Biofuel ethanol'] + 
#                                        Regional_CO2_Budget['Biofuel biodiesal'] + 
#                                        Regional_CO2_Budget['Incineration'] + 
#                                        Regional_CO2_Budget['FF and IPPU'] + 
#                                        Regional_CO2_Budget['Crop yield'] + 
#                                        Regional_CO2_Budget['Livestock Respiration'] + 
#                                        Regional_CO2_Budget['Human Respiration'] + 
#                                        Regional_CO2_Budget['Forest Harvest'] + 
#                                        Regional_CO2_Budget['Forest inventory'] + 
#                                        Regional_CO2_Budget['grassland stockchange'] + 
#                                        Regional_CO2_Budget['cropland stockchange'] +
#                                        Regional_CO2_Budget['residual wood'] +
#                                        Regional_CO2_Budget['PIC and SWDS stockchange'] +
#                                        Regional_CO2_Budget['wood trade'] +
#                                        Regional_CO2_Budget['residual crop'] +
#                                        Regional_CO2_Budget['crop landfill stockchange'] +
#                                        Regional_CO2_Budget['crop trade'] +
#                                        Regional_CO2_Budget['Lake and River emissions'] +
#                                        Regional_CO2_Budget['Lake and River carbon burial'] +
#                                        Regional_CO2_Budget['Coastal carbon export'] ) 
Regional_CO2_Budget['Bottom-up NCE'] = (Regional_CO2_Budget['FF and IPPU'] + 
                                        Regional_CO2_Budget['DeltaC_total'] + 
                                        Regional_CO2_Budget['Crop Harvest (adjusted)'] + 
                                        Regional_CO2_Budget['Forest Harvest (adjusted)'] + 
                                        Regional_CO2_Budget['Respiration'] + 
                                        Regional_CO2_Budget['Biofuel'] + 
                                        Regional_CO2_Budget['Trade'] + 
                                        Regional_CO2_Budget['Residual'] + 
                                        Regional_CO2_Budget['Coastal carbon export'] )

#Regional_CO2_Budget['Bottom-up dC'] = ( Regional_CO2_Budget['Forest inventory'] + 
#                                        Regional_CO2_Budget['grassland stockchange'] + 
#                                        Regional_CO2_Budget['cropland stockchange'] +
#                                        Regional_CO2_Budget['PIC and SWDS stockchange'] +
#                                        Regional_CO2_Budget['crop landfill stockchange'] +
#                                        Regional_CO2_Budget['Lake and River carbon burial'] )
Regional_CO2_Budget['Bottom-up dC'] = (Regional_CO2_Budget['DeltaC_total'] )

#Regional_CO2_Budget['Respiration'] = ( Regional_CO2_Budget['Livestock Respiration'] +
#                                    Regional_CO2_Budget['Human Respiration'] )

#Regional_CO2_Budget['Biofuel'] = ( Regional_CO2_Budget['Biofuel wood'] +
#                                   Regional_CO2_Budget['Biofuel ethanol'] +
#                                   Regional_CO2_Budget['Biofuel biodiesal'] )

#Regional_CO2_Budget['Others'] = ( Regional_CO2_Budget['Incineration'] + 
#                                  Regional_CO2_Budget['residual wood'] +
#                                  Regional_CO2_Budget['wood trade'] +
#                                  Regional_CO2_Budget['residual crop'] +
#                                  Regional_CO2_Budget['crop trade'] +
#                                  Regional_CO2_Budget['Lake and River emissions'] +
#                                  Regional_CO2_Budget['Coastal carbon export'] ) 
Regional_CO2_Budget['Others'] = ( Regional_CO2_Budget['Trade'] +
                                  Regional_CO2_Budget['Residual'] +
                                  Regional_CO2_Budget['Coastal carbon export'] ) 


# Coordinates for sub-plots
latlon_coords = {}
latlon_coords[Region_label[0]] = [-122.5, 43]
latlon_coords[Region_label[1]] = [-107.5, 42]
latlon_coords[Region_label[2]] = [-93.5, 40]
latlon_coords[Region_label[3]] = [-118, 31]
latlon_coords[Region_label[4]] = [-104.5, 30]
latlon_coords[Region_label[5]] = [-89, 32]
latlon_coords[Region_label[6]] = [-79, 40]


# ========== Start Making Plot =========

# Create basemap projection
m = Basemap(width=4800000,height=3050000,resolution='l',projection='laea',lat_ts=39,lat_0=39.,lon_0=(-155-40)/2.)
X,Y = np.meshgrid(lon-0.05/2.,lat-0.05/2.)
xx,yy=m(X,Y)

#Create figure
fig = plt.figure(1, figsize=(6.75*0.95,5.05*0.95), dpi=300)
ax1 = fig.add_axes([0.0,0.163,1.,0.83])

# Create background map
tt = m.pcolormesh(xx,yy,ma.masked_invalid(Regions),alpha=0.4,linewidth=0,rasterized=True,cmap='hsv')
m.drawcoastlines(color='grey',linewidth=0.25)
m.drawcountries(color='grey',linewidth=0.25)
m.drawstates(color='grey',linewidth=0.25)
plt.title('Regional CO$_2$ budget (TgC year$^{-1}$)')

# Create colors
Inv_cmap_test1 = cm.get_cmap('viridis')
Inv_colorst = Inv_cmap_test1(np.linspace(0, 0.9, 5))
Inv_colors = ['grey',Inv_colorst[0],Inv_colorst[1],Inv_colorst[3],Inv_colorst[4],'grey']
colors = [Inv_colors[3],'grey','black','green','orange','saddlebrown','olive','darkslategrey','pink']

# Make inset plots
for Rname in Region_label:
    x1, y1 = m(latlon_coords[Rname][0],latlon_coords[Rname][1])   # get data coordinates for plotting         
    bax = build_bar(x1, y1, ax1, 1.0, [1,2,3,4,5,6,7,8,9],Regional_CO2_Budget.loc[Rname],colors, -200, 370, [-100,0,100,200,300],Rname)

# Create lengend
patch0 = mpatches.Patch(color=colors[0], alpha=0.5, hatch='////', label='NCE (top-down)')
patch1 = mpatches.Patch(color=colors[1], label='NCE (bottom-up)')
patch2 = mpatches.Patch(color=colors[2], label='Fossil fuel')
patch3 = mpatches.Patch(color=colors[3], label='$\mathrm{\Delta C_{Total}}$')
patch4 = mpatches.Patch(color=colors[4], label='Crop harvests$\mathrm{_{adj}}$')
patch5 = mpatches.Patch(color=colors[5], label='Wood harvests$\mathrm{_{adj}}$')
patch6 = mpatches.Patch(color=colors[6], label='Human+Livestock\nrespiration')
patch7 = mpatches.Patch(color=colors[7], label='Biofuels')
patch8 = mpatches.Patch(color=colors[8], label='Others')
ax1.legend(handles=[patch0,patch1,patch2,patch3,patch4,patch5,patch6,patch7,patch8],loc='upper center', bbox_to_anchor=(0.5, 0.012),ncol=3,frameon=False)

# Save figure
plt.savefig('../Figures/Map_regional_TopDown_LNLGIS_vs_bottomup_20240528.png', dpi=300)

