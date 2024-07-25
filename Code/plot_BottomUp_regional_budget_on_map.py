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
test
'''
def build_bar(mapx, mapy, ax, width, xvals, rf, fcolors, ymin, ymax, ytickvec, Rlabel):
    ax_h = inset_axes(ax, width=width*1.1, \
                    height=width*1.1*0.8, \
                    loc=3, \
                    bbox_to_anchor=(mapx, mapy), \
                    bbox_transform=ax.transData, \
                    borderpad=0, \
                    axes_kwargs={'alpha': 0.35, 'visible': True})
    #                                                                                              
    xvals_float = np.asarray(xvals, dtype=np.float32)
    plt.plot([np.min(xvals_float-width)-width, np.max(xvals_float+width)+width],[0,0],'k',linewidth=0.5)
    #plt.plot([(xvals_float[0]-width+xvals_float[1])/2.,(xvals_float[0]-width+xvals_float[1])/2.],[ymin,ymax],'k:',linewidth=1.0)
    plt.fill_between([(xvals_float[0]+xvals_float[1]+width)/2.,np.max(xvals_float+width)+width],[ymin,ymin],[ymax,ymax],color='grey',alpha=0.18)
    #                                                                                              
    yvals = [ rf['Bottom-up NCE'],
              rf['FF and IPPU'],
              rf['Bottom-up dC'],
              rf['Crop yield'],
              rf['Forest Harvest'],
              rf['Respiration'],
              rf['Biofuel'],
              rf['Others'] ]
    #    
    bar_iter = 1
    for x,y,c in zip(xvals, yvals, fcolors):
        #                                                                                          
        #if bar_iter == 0:
        #    ax_h.bar(x-width, y, yerr=yerr, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5,hatch='////',alpha=0.5)
        #    plt.text(x-width,ymin+(ymax-ymin)*0.005,'TD',ha='center',va='bottom',fontsize=6.5)
        #    bar_iter = 1
        if bar_iter == 1:
            ax_h.bar(x, y, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5)
            #plt.text(x-width*2.25/3.,ymin+(ymax-ymin)*0.005,'Bottom-Up',ha='left',va='bottom',fontsize=6.5)
            bar_iter = 2
        else:
            ax_h.bar(x+width, y, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5)
    plt.arrow(np.max(xvals_float+width)+width*2./3.,50,0,100,width=0.07,head_width=0.4,head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))),fc='k')
    plt.text(np.max(xvals_float+width)+width*1.6/3.,100,'source',ha='right',va='center',fontsize=6.5)
    plt.arrow(np.max(xvals_float+width)+width*2./3.,-50,0,-100,width=0.07,head_width=0.4,head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))),fc='k')
    plt.text(np.max(xvals_float+width)+width*1.6/3.,-100,'sink',ha='right',va='center',fontsize=6.5)
    plt.xlim([np.min(xvals_float-width), np.max(xvals_float+width)+width])
    plt.text(np.min(xvals_float-width)-width + ((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))*0.99,ymin+(ymax-ymin)*0.99,Rlabel,ha='right',va='top',fontsize=7)
    plt.xticks([])#range(len(xvals)), xvals, fontsize=10, rotation=30)                             
    ax_h.set_xticklabels([])
    plt.ylim([ymin,ymax])
    plt.yticks(ytickvec)
    ax_h.axis('on')
    return ax_h


def build_bar_CONUS(mapx, mapy, ax, width, xvals, rf, fcolors, ymin, ymax, ytickvec, Rlabel):
    ax_h = inset_axes(ax, width=width*1.1, \
                    height=width*1.1*0.8, \
                    loc=3, \
                    bbox_to_anchor=(mapx, mapy), \
                    bbox_transform=ax.transData, \
                    borderpad=0, \
                    axes_kwargs={'alpha': 0.35, 'visible': True})
    #                                                                                              
    xvals_float = np.asarray(xvals, dtype=np.float32)
    plt.plot([np.min(xvals_float-width)-width, np.max(xvals_float+width)+width],[0,0],'k',linewidth=0.5)
    #plt.plot([(xvals_float[0]-width+xvals_float[1])/2.,(xvals_float[0]-width+xvals_float[1])/2.],[ymin,ymax],'k:',linewidth=1.0)
    plt.fill_between([(xvals_float[0]+xvals_float[1]+width)/2.,np.max(xvals_float+width)+width],[-2000,-2000],[2000,2000],color='grey',alpha=0.18)
    #                                                                                              
    yvals = [ rf['Bottom-up NCE'],
              rf['FF and IPPU'],
              rf['Bottom-up dC'],
              rf['Crop yield'],
              rf['Forest Harvest'],
              rf['Respiration'],
              rf['Biofuel'],
              rf['Others'] ]
    #    
    bar_iter = 1
    for x,y,c in zip(xvals, yvals, fcolors):
        #                                                                                          
        #if bar_iter == 0:
        #    ax_h.bar(x-width, y, yerr=yerr, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5,hatch='////',alpha=0.5)
        #    plt.text(x-width,ymin+(ymax-ymin)*0.005,'TD',ha='center',va='bottom',fontsize=6.5)
        #    bar_iter = 1
        if bar_iter == 1:
            ax_h.bar(x, y, width=1.1, fc=c,capsize=2,edgecolor='k',linewidth=0.5)
            #plt.text(x-width*2.25/3.,ymin+(ymax-ymin)*0.005,'Bottom-Up',ha='left',va='bottom',fontsize=6.5)
            bar_iter = 2
        else:
            ax_h.bar(x+width, y, width=1.0, fc=c,capsize=2,edgecolor='k',linewidth=0.5)
    plt.arrow(np.max(xvals_float+width)+width*2./3.,87,0,310,width=0.07,head_width=0.4,head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))),fc='k')
    plt.text(np.max(xvals_float+width)+width*1.6/3.,270,'source',ha='right',va='center',fontsize=6.5)
    plt.arrow(np.max(xvals_float+width)+width*2./3.,-40,0,-270,width=0.07,head_width=0.4,head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))),fc='k')
    plt.text(np.max(xvals_float+width)+width*1.6/3.,-200,'sink',ha='right',va='center',fontsize=6.5)
    plt.xlim([np.min(xvals_float-width), np.max(xvals_float+width)+width])
    plt.text(np.min(xvals_float-width)-width + ((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))*0.99,ymin+(ymax-ymin)*0.99,Rlabel,ha='right',va='top',fontsize=7)
    plt.xticks([])#range(len(xvals)), xvals, fontsize=10, rotation=30)                             
    ax_h.set_xticklabels([])
    plt.ylim([ymin,ymax])
    plt.yticks(ytickvec)
    ax_h.set_yticklabels(['-400','','','','0','','','','400','','','','800','','','','1200','',''])
    ax_h.axis('on')
    return ax_h


#Create figure
fig = plt.figure(1, figsize=(6.75*0.95,7.05*0.95), dpi=300)


Forest_inventory = pd.read_csv('../Data_processed/Regional_CO2_Budget_Forests.csv')
CropGrass_inventory = pd.read_csv('../Data_processed/Regional_CO2_Budget_CropGrass.csv')
River_inventory = pd.read_csv('../Data_processed/Regional_CO2_Budget_River.csv')
FF_and_IPPU_inventory = pd.read_csv('../Data_processed/Regional_CO2_Budget_Fossil.csv')
summary = pd.read_csv('../Data_processed/Regional_CO2_Budget_Summary.csv')

#Regional_CO2_Budget = pd.read_csv('../Data_processed/Regional_CO2_Budget.csv')

#CONUS_CO2_Budget = Regional_CO2_Budget.sum(axis=0)

#CONUS_CO2_Budget_total = ( CONUS_CO2_Budget['Forest inventory'] + CONUS_CO2_Budget['grassland stockchange'] +
#                           CONUS_CO2_Budget['cropland stockchange'] + CONUS_CO2_Budget['PIC and SWDS stockchange'] +
#                           CONUS_CO2_Budget['crop landfill stockchange'] + CONUS_CO2_Budget['Lake and River carbon burial'])

#print('------')
#print('Total')
#print(CONUS_CO2_Budget_total)
#print('Forest')
#print(CONUS_CO2_Budget['Forest inventory'])
#print('PIC and SWDS stockchange')
#print(CONUS_CO2_Budget['PIC and SWDS stockchange'])
#print('Lake and River carbon burial')
#print(CONUS_CO2_Budget['Lake and River carbon burial'])
#print('Agg + grass')
#print(CONUS_CO2_Budget['grassland stockchange']+CONUS_CO2_Budget['cropland stockchange'])
#print('------')

Inv_cmap_test1 = cm.get_cmap('viridis')
Inv_colorst = Inv_cmap_test1(np.linspace(0, 1.0, 7))


ax1 = fig.add_axes([0.15,0.81,0.825,0.17])
plt.fill_between([0,1],[0,0],[summary['Stockchange'],summary['Stockchange']],color='green', edgecolor='black',linewidth=0.5)
plt.fill_between([1,2],[0,0],[Forest_inventory['stockchange'],Forest_inventory['stockchange']],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
plt.fill_between([2,3],[0,0],[CONUS_CO2_Budget['grassland stockchange'],CONUS_CO2_Budget['grassland stockchange']],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
plt.fill_between([3,4],[0,0],[CONUS_CO2_Budget['cropland stockchange'],CONUS_CO2_Budget['cropland stockchange']],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
plt.fill_between([4,5],[0,0],[CONUS_CO2_Budget['PIC and SWDS stockchange'],CONUS_CO2_Budget['PIC and SWDS stockchange']],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
plt.fill_between([5,6],[0,0],[CONUS_CO2_Budget['crop landfill stockchange'],CONUS_CO2_Budget['crop landfill stockchange']],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
plt.fill_between([6,7],[0,0],[CONUS_CO2_Budget['Lake and River carbon burial'],CONUS_CO2_Budget['Lake and River carbon burial']],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
plt.xlim([-0.7,7.25])
plt.xticks(np.arange(7)+0.5)
ax1.set_xticklabels(['$\mathrm{ \Delta C_{Total} }$',
                     '$\mathrm{ \Delta C_{Forest} }$',
                     '$\mathrm{ \Delta C_{Grassland} }$',
                     '$\mathrm{ \Delta C_{Cropland} }$',
                     'PIC and SWDS',
                     'landfill crops',
                     'aquatic burial'],rotation=20)
plt.ylim([-350,25])
plt.arrow(-0.575,-25,0,-300,width=0.0015,head_width=0.1,head_length=16,fc='k')
plt.text(-0.59,(-25.-325.)/2.,'increasing\nstocks',rotation=270,ha='left',va='center')
plt.ylabel('carbon stockchange\n(TgC year$^{-1}$)')
plt.plot([-0.7,7.25],[0,0],'k',linewidth=0.3)
plt.text(7,-60,'(a)',ha='right',va='top',fontsize=12)



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

# Coordinates for sub-plots
latlon_coords = {}
latlon_coords[Region_label[0]] = [-122.9, 43]
latlon_coords[Region_label[1]] = [-107.9, 42]
latlon_coords[Region_label[2]] = [-93.9, 40]
latlon_coords[Region_label[3]] = [-119, 34]
latlon_coords[Region_label[4]] = [-104., 32.5]
latlon_coords[Region_label[5]] = [-89, 32]
latlon_coords[Region_label[6]] = [-79.5, 40]


# ========== Start Making Plot =========

# Create basemap projection
m = Basemap(width=4800000,height=3050000,resolution='l',projection='laea',lat_ts=39,lat_0=39.,lon_0=(-155-40)/2.)
X,Y = np.meshgrid(lon-0.05/2.,lat-0.05/2.)
xx,yy=m(X,Y)


ax1 = fig.add_axes([0.015,0.005,0.97,0.83])

# Create background map
tt = m.pcolormesh(xx,yy,ma.masked_invalid(Regions),alpha=0.4,linewidth=0,rasterized=True,cmap='hsv')
m.drawcoastlines(color='grey',linewidth=0.25)
m.drawcountries(color='grey',linewidth=0.25)
m.drawstates(color='grey',linewidth=0.25)
#plt.title('Regional CO$_2$ budget (TgC year$^{-1}$)')

# Create colors
Inv_cmap_test1 = cm.get_cmap('viridis')
Inv_colorst = Inv_cmap_test1(np.linspace(0, 0.9, 5))
Inv_colors = ['grey',Inv_colorst[0],Inv_colorst[1],Inv_colorst[3],Inv_colorst[4],'grey']
colors = ['grey','black','green','orange','saddlebrown','olive','darkslategrey','pink']

# Make inset plots
for Rname in Region_label:
    x1, y1 = m(latlon_coords[Rname][0],latlon_coords[Rname][1])   # get data coordinates for plotting         
    bax = build_bar(x1, y1, ax1, 1.0, [2,3,4,5,6,7,8,9],Regional_CO2_Budget.loc[Rname],colors, -200, 370, [-100,0,100,200,300],Rname)

CONUS_total = Regional_CO2_Budget.sum()
x1, y1 = m(-117.5,24)
bax = build_bar_CONUS(x1, y1, ax1, 1.0, [2,3,4,5,6,7,8,9],CONUS_total,colors, -400, 1500, [-400,-300,-200,-100,0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400],'CONUS total')

text_x, text_y = m(-66.1, 47.3)
ax1.text(text_x, text_y, '(b)', ha='right', va='top', fontsize=12)

# Create lengend
#patch0 = mpatches.Patch(color=colors[0], alpha=0.5, hatch='////', label='NCE (top-down)')
patch1 = mpatches.Patch(color=colors[0], label='NCE (bottom-up)')
patch2 = mpatches.Patch(color=colors[1], label='Fossil fuel')
patch3 = mpatches.Patch(color=colors[2], label='$\mathrm{\Delta C_{Total}}$')
patch4 = mpatches.Patch(color=colors[3], label='Crop harvests')
patch5 = mpatches.Patch(color=colors[4], label='Wood harvests')
patch6 = mpatches.Patch(color=colors[5], label='Human+Livestock\nrespiration')
patch7 = mpatches.Patch(color=colors[6], label='Biofuels')
patch8 = mpatches.Patch(color=colors[7], label='Others')
ax1.legend(handles=[patch1,patch2,patch3,patch4,patch5,patch6,patch7,patch8],loc='upper center', bbox_to_anchor=(0.5, 0.018),ncol=3,frameon=False)

# Save figure
plt.savefig('../Figures/Map_regional_bottomup_20240617.png', dpi=300)

