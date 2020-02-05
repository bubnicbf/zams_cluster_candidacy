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


def make_final_plot(fig_num, m_rang, zam_met, metals_z, metals_feh,
                    final_zams_params, final_zams, out_dir):
    '''
    Print the final plot with all the sequences superimposed and fitted by a
    polynomial.
    '''

    # Store min and max metallicity ranges.
    metal_min, metal_max = m_rang[0], m_rang[1]
        
    # Store ages in array.
    ages, names, names_feh = [], [], []
    for seq_param in final_zams_params:
        if metal_min <= seq_param[3] <= metal_max:
            ages.append(seq_param[2])
            names.append(seq_param[0])
            names_feh.append(seq_param[0]+' ('+str(seq_param[3])+')')
    ages = np.array(ages)

    # Skip if no sequences are inside this metallicity range.
    if len(ages) > 0:

        # Create new interpolated list.
        final_zams_poli = []
        for indx, seq in enumerate(final_zams):
            if metal_min <= final_zams_params[indx][3] <= metal_max:
                # Obtain and plot fitting polinome.
                poli = np.polyfit(seq[1], seq[0], 3)
                y_pol = np.linspace(min(seq[1]), max(seq[1]), 50).tolist()
                p = np.poly1d(poli)
                x_pol = [p(i) for i in y_pol]
                final_zams_poli.append([x_pol, y_pol])    

        # Obtain plotting limits.        
        min_lim = np.min(np.min(final_zams_poli, axis=2) ,axis=0)
        max_lim = np.max(np.max(final_zams_poli, axis=2) ,axis=0)      
        
        # Sort all lists according to age.
        ages_s, names_s, names_feh_s, final_zams_poli_s = \
        map(list, zip(*sorted(zip(ages, names, names_feh, final_zams_poli),
                              reverse=True)))
        
        # figsize(x1, y1), GridSpec(y2, x2) -> To have square plots:
        # x1/x2 = y1/y2 = 2.5 
        fig = plt.figure(figsize=(45, 25)) # create the top-level container
        gs = gridspec.GridSpec(10, 30)  # create a GridSpec object
    
    
        ax1 = plt.subplot(gs[1:9, 0:11])   
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
        # Plot colorbar.
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(ages_s)
        cbar = plt.colorbar(m)
        cbar.set_label(r'Age (Gyr)', fontsize=26, labelpad=20)
        cbar.ax.tick_params(which='major', length=12, labelsize=24)
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
                 
                
                
        ax2 = plt.subplot(gs[1:9, 11:20])    
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
        # Rearrange sequences into single list composed of two sub-lists: the
        # first one holds the colors and the second one the magnitudes.
        single_seq_list = [[i for v in r for i in v] for r in \
        zip(*final_zams_poli_s)]
        # Obtain and plot fitting polinome for all sequences.
        poli_order = [3]
        pol_col = ['r', 'b']
        for j, order in enumerate(poli_order):
            poli_zams = np.polyfit(single_seq_list[1], single_seq_list[0], order)
            y_pol = np.linspace(min(single_seq_list[1]),
                                max(single_seq_list[1]), 50)
            p = np.poly1d(poli_zams)
            x_pol = [p(i) for i in y_pol]
            plt.plot(x_pol, y_pol, c=pol_col[j], lw=2.5,
                     label='ZAMS (%d)' % order)
        # Plot ZAMS envelope.
        k = 1 if min_met == max_met else 2
        for j in range(k):
            text1 = 'z = %0.3f' '\n' % metals_z[a[j]]
            text2 = '[Fe/H] = %0.2f' % metals_feh[a[j]]
            text = text1+text2
            plt.plot(zam_met[a[j]][3], zam_met[a[j]][2], c='k', ls='--',
                     lw=1.5, label=text)    
        # Add legend.
        leg = ax2.legend(loc="upper right", markerscale=1.5, scatterpoints=2,
                   fontsize=18)
        leg.get_frame().set_alpha(0.5)
        
        
        ax3 = plt.subplot(gs[1:9, 20:29])
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
        # Rearrange sequences into single list composed of two sub-lists: the
        # first one holds the colors and the second one the magnitudes.
        single_seq_list = [[i for v in r for i in v] for r in \
        zip(*final_zams_poli_s)]
        # Obtain and plot fitting polinome for all sequences.
        poli_order = [3]
        pol_col = ['r', 'b']
        for j, order in enumerate(poli_order):
            poli_zams = np.polyfit(single_seq_list[1], single_seq_list[0], order)
            y_pol = np.linspace(min(single_seq_list[1]),
                                max(single_seq_list[1]), 50)
            p = np.poly1d(poli_zams)
            x_pol = [p(i) for i in y_pol]
            plt.plot(x_pol, y_pol, c=pol_col[j], lw=2.5,
                     label='ZAMS (%d)' % order)
        # Plot ZAMS envelope.
        k = 1 if min_met == max_met else 2
        for j in range(k):
            text1 = 'z = %0.3f' '\n' % metals_z[a[j]]
            text2 = '[Fe/H] = %0.2f' % metals_feh[a[j]]
            text = text1+text2
            plt.plot(zam_met[a[j]][3], zam_met[a[j]][2], c='k', ls='--',
                     lw=1.5, label=text)    
        # Add legend.
        leg = ax3.legend(loc="upper right", markerscale=1.5, scatterpoints=2,
                   fontsize=18)
        leg.get_frame().set_alpha(0.5)
                   
        
        
        fig.tight_layout()
        # Generate output file for each data file.
        plt.savefig(join(out_dir+'fitted_zams/'+'final_ZAMS_%s.png' % fig_num),
                    dpi=150)

    else:
        print 'Skipped %d' % fig_num
        