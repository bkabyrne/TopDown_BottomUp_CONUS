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
Plot Figure 1 for paper
'''

def build_bar(mapx, mapy, ax, width, xvals, rf, fcolors, ymin, ymax, ytickvec, Rlabel, aggregation = 'Regional'):
    """
    Make bar plot of carbon budget for CONUS region.
    """
    # Create an inset axis for the bar plot
    ax_h = inset_axes(ax, width=width*1.1,
                      height=width*1.1*0.8,
                      loc=3,
                      bbox_to_anchor=(mapx, mapy),
                      bbox_transform=ax.transData,
                      borderpad=0,
                      axes_kwargs={'alpha': 0.35, 'visible': True})

    # Convert xvals to float32 and create the baseline line
    xvals_float = np.asarray(xvals, dtype=np.float32)
    plt.plot([np.min(xvals_float-width)-width, np.max(xvals_float+width)+width], [0, 0], 'k', linewidth=0.5)
    plt.fill_between([(xvals_float[0]+xvals_float[1]+width)/2., np.max(xvals_float+width)+width], [ymin, ymin], [ymax, ymax], color='grey', alpha=0.18)

    # Define the y-values for the bars
    yvals = [
        rf['NCE'],
        rf['FF and IPPU'],
        rf['$\Delta$C_{Total}'],
        rf['Crop Harvest_{adj}'],
        rf['Forest Harvest_{adj}'],
        rf['Respiration'],
        rf['Biofuel'],
        rf['Others']
    ]

    # Plot the bars
    bar_iter = 1
    for x, y, c in zip(xvals, yvals, fcolors):
        if bar_iter == 1:
            ax_h.bar(x, y, width=1.0, fc=c, capsize=2, edgecolor='k', linewidth=0.5)
            bar_iter = 2
        else:
            ax_h.bar(x+width, y, width=1.0, fc=c, capsize=2, edgecolor='k', linewidth=0.5)

    # Draw arrows and labels for sources and sinks
    if aggregation == 'Regional':
        plt.arrow(np.max(xvals_float+width)+width*2./3., 50, 0, 100, width=0.07, head_width=0.4, 
                  head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))), fc='k')
        plt.text(np.max(xvals_float+width)+width*1.6/3., 100, 'source', ha='right', va='center', fontsize=6.5)
        plt.arrow(np.max(xvals_float+width)+width*2./3., -50, 0, -100, width=0.07, head_width=0.4, 
                  head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))), fc='k')
        plt.text(np.max(xvals_float+width)+width*1.6/3., -100, 'sink', ha='right', va='center', fontsize=6.5)
    elif aggregation == 'CONUS':
        plt.arrow(np.max(xvals_float+width)+width*2./3., 87, 0, 310, width=0.07, head_width=0.4, 
                  head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))), fc='k')
        plt.text(np.max(xvals_float+width)+width*1.6/3., 270, 'source', ha='right', va='center', fontsize=6.5)
        plt.arrow(np.max(xvals_float+width)+width*2./3., -40, 0, -270, width=0.07, head_width=0.4, 
                  head_length=0.6 * ((ymax-ymin)/((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))), fc='k')
        plt.text(np.max(xvals_float+width)+width*1.6/3., -200, 'sink', ha='right', va='center', fontsize=6.5)

    # Set axis limits and labels
    plt.xlim([np.min(xvals_float-width), np.max(xvals_float+width)+width])
    plt.text(np.min(xvals_float-width)-width + ((np.max(xvals_float+width)+width) - (np.min(xvals_float-width)-width))*0.99, ymin+(ymax-ymin)*0.99, Rlabel, ha='right', va='top', fontsize=7)
    plt.xticks([])
    ax_h.set_xticklabels([])
    plt.ylim([ymin, ymax])
    plt.yticks(ytickvec)
    ax_h.axis('on')
    if aggregation == 'CONUS':
        ax_h.set_yticklabels(['-400','','','','0','','','','400','','','','800','','','','1200','',''])

    return ax_h


def make_plot(Forest_inventory,CropGrass_inventory,River_inventory,summary,figure_name):

    #Create figure
    fig = plt.figure(1, figsize=(6.75*0.95,7.05*0.95), dpi=300)

    # ======== PANEL A ======== 

    ax1 = fig.add_axes([0.15,0.81,0.825,0.17])

    Inv_cmap_test1 = matplotlib.colormaps['viridis']
    Inv_colorst = Inv_cmap_test1(np.linspace(0, 1.0, 7))

    plt.fill_between([0,1],[0,0],[summary['$\Delta$C_{Total}'].sum(),summary['$\Delta$C_{Total}'].sum()],color='green', edgecolor='black',linewidth=0.5)
    plt.fill_between([1,2],[0,0],[Forest_inventory['DeltaC_forest'].sum(),Forest_inventory['DeltaC_forest'].sum()],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
    plt.fill_between([2,3],[0,0],[CropGrass_inventory['DeltaC_grassland'].sum(),CropGrass_inventory['DeltaC_grassland'].sum()],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
    plt.fill_between([3,4],[0,0],[CropGrass_inventory['DeltaC_cropland'].sum(),CropGrass_inventory['DeltaC_cropland'].sum()],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
    plt.fill_between([4,5],[0,0],[Forest_inventory['PIC and SWDS'].sum(),Forest_inventory['PIC and SWDS'].sum()],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
    plt.fill_between([5,6],[0,0],[CropGrass_inventory['landfill crops'].sum(),CropGrass_inventory['landfill crops'].sum()],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
    plt.fill_between([6,7],[0,0],[River_inventory['aquatic burial'].sum(),River_inventory['aquatic burial'].sum()],color='green', edgecolor='black',linewidth=0.5,alpha=0.4)
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


    # ======== PANEL B ======== 

    ax1 = fig.add_axes([0.015,0.005,0.97,0.83])
    
    # Create basemap projection
    lat, lon, Regions, area = utils.Regional_mask('005')
    Regions[np.where(Regions==0)]=np.nan
    m = Basemap(width=4800000,height=3050000,resolution='l',projection='laea',lat_ts=39,lat_0=39.,lon_0=(-155-40)/2.)
    X,Y = np.meshgrid(lon-0.05/2.,lat-0.05/2.)
    xx,yy=m(X,Y)
    
    # Create background map
    tt = m.pcolormesh(xx,yy,ma.masked_invalid(Regions),alpha=0.4,linewidth=0,rasterized=True,cmap='hsv')
    m.drawcoastlines(color='grey',linewidth=0.25)
    m.drawcountries(color='grey',linewidth=0.25)
    m.drawstates(color='grey',linewidth=0.25)
    #plt.title('Regional CO$_2$ budget (TgC year$^{-1}$)')
    
    # Create colors
    Inv_colorst = Inv_cmap_test1(np.linspace(0, 0.9, 5))
    Inv_colors = ['grey',Inv_colorst[0],Inv_colorst[1],Inv_colorst[3],Inv_colorst[4],'grey']
    colors = ['grey','black','green','orange','saddlebrown','olive','darkslategrey','pink']

    # Where to plot inserts
    Region_label = ['Northwest','N Great Plains','Midwest','Southwest','S Great Plains','Southeast','Northeast']
    # Coordinates for sub-plots
    latlon_coords = {}
    latlon_coords[Region_label[0]] = [-122.9, 43]
    latlon_coords[Region_label[1]] = [-107.9, 42]
    latlon_coords[Region_label[2]] = [-93.9, 40]
    latlon_coords[Region_label[3]] = [-119, 34]
    latlon_coords[Region_label[4]] = [-104., 32.5]
    latlon_coords[Region_label[5]] = [-89, 32]
    latlon_coords[Region_label[6]] = [-79.5, 40]

    # Make inset plots
    for Rname in Region_label:
        x1, y1 = m(latlon_coords[Rname][0],latlon_coords[Rname][1])   # get data coordinates for plotting         
        bax = build_bar(x1, y1, ax1, 1.0, [2,3,4,5,6,7,8,9],summary.loc[Rname],colors, -200, 370, [-100,0,100,200,300],Rname)

    # Make CONUS total inset
    CONUS_total = summary.sum()
    x1, y1 = m(-117.5,24)
    bax = build_bar(x1, y1, ax1, 1.0, [2,3,4,5,6,7,8,9],CONUS_total,colors, -400, 1500, [-400,-300,-200,-100,0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400],'CONUS total',aggregation = 'CONUS')

    # add panel label
    text_x, text_y = m(-66.1, 47.3)
    ax1.text(text_x, text_y, '(b)', ha='right', va='top', fontsize=12)
    
    # Create lengend
    patch1 = mpatches.Patch(color=colors[0], label='NCE')
    patch2 = mpatches.Patch(color=colors[1], label='Fossil fuel')
    patch3 = mpatches.Patch(color=colors[2], label='$\mathrm{\Delta C_{Total}}$')
    patch4 = mpatches.Patch(color=colors[3], label='Crop $\mathrm{harvests_{adj}}$')
    patch5 = mpatches.Patch(color=colors[4], label='Wood $\mathrm{harvests_{adj}}$')
    patch6 = mpatches.Patch(color=colors[5], label='Human+Livestock\nrespiration')
    patch7 = mpatches.Patch(color=colors[6], label='Biofuels')
    patch8 = mpatches.Patch(color=colors[7], label='Others')
    ax1.legend(handles=[patch1,patch2,patch3,patch4,patch5,patch6,patch7,patch8],loc='upper center', bbox_to_anchor=(0.5, 0.018),ncol=3,frameon=False)
    
    # Save figure
    plt.savefig(figure_name, dpi=300)

    
def main():
    
    # Prepare data
    Forest_inventory = pd.read_csv('../Data_processed/Regional_CO2_Budget_Forests.csv')
    CropGrass_inventory = pd.read_csv('../Data_processed/Regional_CO2_Budget_CropGrass.csv')
    River_inventory = pd.read_csv('../Data_processed/Regional_CO2_Budget_River.csv')
    summary = pd.read_csv('../Data_processed/Regional_CO2_Budget_Summary.csv')

    # Add and re-label columns
    numeric_cols = summary.select_dtypes(include=[np.number]).columns
    summary['NCE'] = summary[numeric_cols].sum(axis=1)
    summary.rename(columns={'DeltaC_total': '$\Delta$C_{Total}'}, inplace=True)
    summary['Others'] = ( summary['Trade'] +
                          summary['Residual'] +
                          summary['Coastal carbon export'] )
    summary['Crop Harvest_{adj}'] = CropGrass_inventory['Harvest (adjusted)']
    summary['Forest Harvest_{adj}'] = Forest_inventory['Harvest (adjusted)']
    summary.set_index('Region', inplace=True)

    # Make plot
    figure_name = '../Figures/Map_regional_bottomup_20240617.png'
    make_plot(Forest_inventory,CropGrass_inventory,River_inventory,summary,figure_name)

if __name__ == "__main__":
    main()
