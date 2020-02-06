# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:03:48 2013

@author: gabriel
"""

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator


def make_final_plot(fig_num, ages_s, names_s, names_feh_s, final_zams_poli_s,
                    metal_min, metal_max, zam_met, metals_z, metals_feh,
                    zx_pol, zy_pol, out_dir):
    '''
    Print the final plot with all the sequences superimposed and fitted by a
    polynomial.
    '''

        
    # Obtain plotting limits.        
    min_lim = np.min(np.min(final_zams_poli_s, axis=2) ,axis=0)
    max_lim = np.max(np.max(final_zams_poli_s, axis=2) ,axis=0)      
    
    
    # figsize(x1, y1), GridSpec(y2, x2) -> To have square plots:
    # x1/x2 = y1/y2 = 2.5 
    fig = plt.figure(figsize=(45, 25)) # create the top-level container
    gs = gridspec.GridSpec(10, 30)  # create a GridSpec object


    ax1 = plt.subplot(gs[1:9, 0:10])   
    plt.xlim(min_lim[0]-1., max_lim[0]+1.)
    plt.ylim(max_lim[1]+0.5, min_lim[1]-0.5)
    plt.xlabel(r'$(C-T_1)_o$', fontsize=28)
    plt.ylabel(r'$M_{T_1}$', fontsize=28)
    # Ticks.
    ax1.xaxis.set_major_locator(MultipleLocator(1.0))
    ax1.minorticks_on()
    ax1.tick_params(which='minor', length=8)
    ax1.tick_params(which='major', length=12)
    ax1.grid(b=True, which='both', color='gray', linestyle='--', lw=1)
    ax1.tick_params(axis='both', which='major', labelsize=26)
    # Plot each cluster's sequence.
    cmap = plt.get_cmap('rainbow')
    k = 0
    for (x, y), color, label in zip(final_zams_poli_s, ages_s, names_feh_s):
        # Transform color value.
        m, h = 1./(max(ages_s) - min(ages_s)), \
        min(ages_s)/(min(ages_s) - max(ages_s))
        col_transf = m*color+h
        l, = plt.plot(x, y, label=label, color=cmap(col_transf), lw=2.)
        pos = [x[-1], y[-1]+0.04]
        plt.text(pos[0], pos[1], names_s[k], size=16, rotation=0,
                 color=l.get_color(), ha="center", va="center",\
                 bbox=dict(ec='1',fc='1', alpha=0.6))        
        k += 1
    # Find ZAMS to plot according to the metallicity range used.
    min_met = min(range(len(metals_feh)), key=lambda i: \
    abs(metals_feh[i]-metal_min))
    max_met = min(range(len(metals_feh)), key=lambda i: \
    abs(metals_feh[i]-metal_max))
    a = [min_met, max_met]
    # Plot ZAMS envelope.
    k = 1 if min_met == max_met else 2
    for j in range(k):
        plt.plot(zam_met[a[j]][3], zam_met[a[j]][2], c='k', ls='--', lw=1.5)
    # Add legend.        
    leg = ax1.legend(loc="upper right", markerscale=1.5, scatterpoints=2,
               fontsize=18)
    leg.get_frame().set_alpha(0.5)
    # Add text box
    if metal_min == metal_max :
        text = r'[Fe/H] $=$ %0.2f' % (metal_min)
    else:
        text = r'%0.2f $\leq$ [Fe/H] $<$ %0.2f' % (metal_min, metal_max )
    plt.text(0.355, 0.975, text, transform=ax1.transAxes,
             bbox=dict(facecolor='gray', alpha=0.1,
                       boxstyle='round,pad=0.4'), fontsize=24)
             
            
            
    ax2 = plt.subplot(gs[1:9, 10:20])    
    plt.xlim(min_lim[0]-1., max_lim[0]+1.)
    plt.ylim(max_lim[1]+0.5, min_lim[1]-0.5)
    plt.xlabel(r'$(C-T_1)_o$', fontsize=28)
    plt.ylabel(r'$M_{T_1}$', fontsize=28)
    # Ticks.
    ax2.xaxis.set_major_locator(MultipleLocator(1.0))
    ax2.minorticks_on()
    ax2.tick_params(which='minor', length=8)
    ax2.tick_params(which='major', length=12)
    ax2.grid(b=True, which='both', color='gray', linestyle='--', lw=1)
    ax2.tick_params(axis='both', which='major', labelsize=26)
    # Plot fitting polinome for all sequences (final ZAMS).
    plt.plot(zx_pol, zy_pol, c='r', lw=2.5, label='ZAMS')
    # Plot ZAMS envelope.
    k = 1 if min_met == max_met else 2
    ls_lst = ['--', '-.']
    for j in range(k):
        text1 = 'z = %0.3f' '\n' % metals_z[a[j]]
        text2 = '[Fe/H] = %0.2f' % metals_feh[a[j]]
        text = text1+text2
        plt.plot(zam_met[a[j]][3], zam_met[a[j]][2], c='k', ls=ls_lst[j],
                 lw=2., label=text)    
    # Add legend.
    leg = ax2.legend(loc="upper right", markerscale=1.5, scatterpoints=2,
               fontsize=18)
    leg.get_frame().set_alpha(0.5)
    
    
    ax3 = plt.subplot(gs[1:9, 20:30])
    plt.xlim(min_lim[0]-1., max_lim[0]+1.)
    plt.ylim(max_lim[1]+0.5, min_lim[1]-0.5)
    plt.xlabel(r'$(C-T_1)_o$', fontsize=28)
    plt.ylabel(r'$M_{T_1}$', fontsize=28)
    # Ticks.
    ax3.xaxis.set_major_locator(MultipleLocator(1.0))
    ax3.minorticks_on()
    ax3.tick_params(which='minor', length=8)
    ax3.tick_params(which='major', length=12)
    ax3.grid(b=True, which='both', color='gray', linestyle='--', lw=1)
    ax3.tick_params(axis='both', which='major', labelsize=26)

    # Plot ZAMS envelope.
    k = 1 if min_met == max_met else 2
    ls_lst = ['--', '-.']
    for j in range(k):
        text1 = 'z = %0.3f' '\n' % metals_z[a[j]]
        text2 = '[Fe/H] = %0.2f' % metals_feh[a[j]]
        text = text1+text2
        plt.plot(zam_met[a[j]][3], zam_met[a[j]][2], c='k', ls=ls_lst[j],
                 lw=2., label=text)    
    # Add legend.
    leg = ax3.legend(loc="upper right", markerscale=1.5, scatterpoints=2,
               fontsize=18)
    leg.get_frame().set_alpha(0.5)
               
    
    
    fig.tight_layout()
    # Generate output file for each data file.
    plt.savefig(join(out_dir+'fitted_zams/'+'final_ZAMS_%s.png' % fig_num),
                dpi=150)
        