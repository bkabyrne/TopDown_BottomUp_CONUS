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
plot_OCO2MIP_experiments_regional_budget_on_map.py

Makes regional NCE plots for all inversion experiments on bottom-up estimate

Output:
  ../Figures/Map_regional_NCE_TopDown_all_vs_bottomup__20240412.png

'''

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
    plt.plot([(xvals_float[4]+xvals_float[5])/2.,(xvals_float[4]+xvals_float[5])/2.],[ymin,ymax],'k:',linewidth=1.0)
    #plt.fill_between([(xvals_float[1]+xvals_float[2]+width)/2.,np.max(xvals_float+width)+width],[ymin,ymin],[ymax,ymax],color='grey',alpha=0.18)
    #
    yvals = [ rf['Prior median'],
              rf['IS median'],
              rf['LNLG median'],
              rf['LNLGIS median'],
              rf['LNLGOGIS median'],
              rf['Bottom-up NCE'] ]

    yerr = [ rf['Prior std'],
             rf['IS std'],
             rf['LNLG std'],
             rf['LNLGIS std'],
             rf['LNLGOGIS std'],
             0]

    bar_iter = 0
    for x,y,ye,c in zip(xvals, yvals, yerr, fcolors):
        #
        print(ye)
        if ye > 0:
            ax_h.bar(x-width, y, yerr=ye, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5,hatch='////',alpha=0.5)
        elif ye == 1:
            ax_h.bar(x, y, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5)
            bar_iter = 2
        else:
            ax_h.bar(x+width, y, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5)
    plt.xlim([np.min(xvals_float-width)-width, np.max(xvals_float+width)+width])
    plt.text(np.min(xvals_float-width)-width + ((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))*0.01,ymin+(ymax-ymin)*0.01,Rlabel,ha='left',va='bottom',fontsize=7)
    plt.xticks([])
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

'''
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
'''
Regional_CO2_Budget['Bottom-up NCE'] = (Regional_CO2_Budget['FF and IPPU'] + 
                                        Regional_CO2_Budget['DeltaC_total'] + 
                                        Regional_CO2_Budget['Crop Harvest (adjusted)'] + 
                                        Regional_CO2_Budget['Forest Harvest (adjusted)'] + 
                                        Regional_CO2_Budget['Respiration'] + 
                                        Regional_CO2_Budget['Biofuel'] + 
                                        Regional_CO2_Budget['Trade'] + 
                                        Regional_CO2_Budget['Residual'] + 
                                        Regional_CO2_Budget['Coastal carbon export'] )

'''
Regional_CO2_Budget['Bottom-up dC'] = ( Regional_CO2_Budget['Forest inventory'] + 
                                        Regional_CO2_Budget['grassland stockchange'] + 
                                        Regional_CO2_Budget['cropland stockchange'] +
                                        Regional_CO2_Budget['PIC and SWDS stockchange'] +
                                        Regional_CO2_Budget['crop landfill stockchange'] +
                                        Regional_CO2_Budget['Lake and River carbon burial'] )
'''
Regional_CO2_Budget['Bottom-up dC'] = (Regional_CO2_Budget['DeltaC_total'] )

'''
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
'''
Regional_CO2_Budget['Others'] = ( Regional_CO2_Budget['Trade'] +
                                  Regional_CO2_Budget['Residual'] +
                                  Regional_CO2_Budget['Coastal carbon export'] ) 

Regional_TopDown_NCE = pd.read_csv('../Data_processed/Regional_TopDown_NCE.csv')
Regional_TopDown_NCE.set_index('region', inplace=True)
Regional_CO2_Budget['Prior median'] = Regional_TopDown_NCE['Prior median'].copy()
Regional_CO2_Budget['Prior std'] = Regional_TopDown_NCE['Prior std'].copy()
Regional_CO2_Budget['LNLG median'] = Regional_TopDown_NCE['LNLG median'].copy()
Regional_CO2_Budget['LNLG std'] = Regional_TopDown_NCE['LNLG std'].copy()
Regional_CO2_Budget['IS median'] = Regional_TopDown_NCE['IS median'].copy()
Regional_CO2_Budget['IS std'] = Regional_TopDown_NCE['IS std'].copy()
Regional_CO2_Budget['LNLGIS median'] = Regional_TopDown_NCE['LNLGIS median'].copy()
Regional_CO2_Budget['LNLGIS std'] = Regional_TopDown_NCE['LNLGIS std'].copy()
Regional_CO2_Budget['LNLGOGIS median'] = Regional_TopDown_NCE['LNLGOGIS median'].copy()
Regional_CO2_Budget['LNLGOGIS std'] = Regional_TopDown_NCE['LNLGOGIS std'].copy()

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


# Make inset plots
for Rname in Region_label:
    x1, y1 = m(latlon_coords[Rname][0],latlon_coords[Rname][1])   # get data coordinates for plotting         
    bax = build_bar(x1, y1, ax1, 1.0, [1,2,3,4,5,6],Regional_CO2_Budget.loc[Rname],Inv_colors, -200, 370, [-100,0,100,200,300],Rname)

# Create lengend
patch0 = mpatches.Patch(color=Inv_colors[0], alpha=0.5, hatch='////', label='Prior')
patch1 = mpatches.Patch(color=Inv_colors[1], alpha=0.5, hatch='////', label='IS')
patch2 = mpatches.Patch(color=Inv_colors[2], alpha=0.5, hatch='////', label='LNLG')
patch3 = mpatches.Patch(color=Inv_colors[3], alpha=0.5, hatch='////', label='LNLGIS')
patch4 = mpatches.Patch(color=Inv_colors[4], alpha=0.5, hatch='////', label='LNLGOGIS')
patch5 = mpatches.Patch(color=Inv_colors[5], label='Bottom-up')

ax1.legend(handles=[patch0,patch1,patch2,patch3,patch4,patch5],loc='upper center', bbox_to_anchor=(0.5, 0.012),ncol=3,frameon=False)

# Save figure
plt.savefig('../Figures/Map_regional_NCE_TopDown_all_vs_bottomup__20240412.png', dpi=300)

