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


def make_final_plot(metal_range, zam_met, metals_z, metals_feh,
                    final_zams_params, final_zams, out_dir):
    '''
    Print the final plot with all the sequences superimposed and fitted by a
    polynomial.
    '''
    
    if metal_range == 0:
        metal_min, metal_max = -1.4, -0.71
    elif metal_range == 1:
        metal_min, metal_max = -0.7, -0.4
    elif metal_range == 2:
        metal_min, metal_max = -0.39, 0.
        
        
    # Store ages in array.
    ages, names, names_feh = [], [], []
    for seq_param in final_zams_params:
        if seq_param[3] >= metal_min and seq_param[3] <= metal_max:
            ages.append(seq_param[2])
            names.append(seq_param[0])
            names_feh.append(seq_param[0]+' ('+str(seq_param[3])+')')
    ages = np.array(ages)

    # Skip if no sequences are inside this metallicity range.
    if len(ages) > 0:

        # Create new interpolated list.
        final_zams_poli = []
        for indx, seq in enumerate(final_zams):
            if final_zams_params[indx][3] >= metal_min and \
            final_zams_params[indx][3] <= metal_max:
                # Obtain and plot fitting polinome.
                poli = np.polyfit(seq[1], seq[0], 3)
                y_pol = np.linspace(min(seq[1]), max(seq[1]), 50)
                p = np.poly1d(poli)
                x_pol = [p(i) for i in y_pol]
                final_zams_poli.append([x_pol, y_pol])    
        
        
        # Sort all lists according to age.
        ages_s, names_s, names_feh_s, final_zams_poli_s = \
        map(list, zip(*sorted(zip(ages, names, names_feh, final_zams_poli),
                              reverse=True)))
        
        # figsize(x1, y1), GridSpec(y2, x2) -> To have square plots:
        # x1/x2 = y1/y2 = 2.5 
        fig = plt.figure(figsize=(40, 25)) # create the top-level container
        gs = gridspec.GridSpec(10, 16)  # create a GridSpec object
    
        ax1 = plt.subplot(gs[1:9, 1:8])    
        plt.ylim(9, -3)
        plt.xlim(-1., 3.)
        plt.xlabel(r'$(C-T_1)_o$', fontsize=28)
        plt.ylabel(r'$M_{T_1}$', fontsize=28)
        # Ticks.
        ax1.xaxis.set_major_locator(MultipleLocator(1.0))
        ax1.minorticks_on()
        ax1.tick_params(which='minor', length=8)
        ax1.tick_params(which='major', length=12)
        ax1.grid(b=True, which='both', color='gray', linestyle='--', lw=1)
        ax1.tick_params(axis='both', which='major', labelsize=26)
        
        cmap = plt.get_cmap('rainbow')
        k = 0
        for (x, y), color, label in zip(final_zams_poli_s, ages_s, names_feh_s):
            # Transform color value.
            m, h = 1./(max(ages_s) - min(ages_s)), \
            min(ages_s)/(min(ages_s) - max(ages_s))
            col_transf = m*color+h
            l, = plt.plot(x, y, label=label, color=cmap(col_transf), lw=2.)
            pos = [(x[-2]+x[-1])/2.+0.15, (y[-2]+y[-1])/2.]
            pos = [x[-1], y[-1]+0.1]
            plt.text(pos[0], pos[1], names_s[k], size=16, rotation=0,
                     color=l.get_color(), ha="center", va="center",\
                     bbox=dict(ec='1',fc='1', alpha=0.6))        
            k += 1
            
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(ages_s)
        cbar = plt.colorbar(m)
        cbar.set_label(r'Age (Gyr)', fontsize=26, labelpad=20)
        cbar.ax.tick_params(which='major', length=12, labelsize=24)
        
        # Add legend.        
        ax1.legend(loc="upper right", markerscale=1.5, scatterpoints=2,
                   fontsize=16)
        # Add text box
        text = r'%0.2f $\leq$ [Fe/H] $\leq$ %0.2f' % (metal_min, metal_max )
        plt.text(0.355, 0.975, text, transform=ax1.transAxes,
                 bbox=dict(facecolor='gray', alpha=0.1,
                           boxstyle='round,pad=0.4'), fontsize=24)
                 
                
                
        ax2 = plt.subplot(gs[1:9, 9:15])    
        plt.ylim(9, -3)
        plt.xlim(-1., 3.)
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
        lin_st = ['--', '-']
        if metal_range == 0:
            a = [0, 1]
        elif metal_range == 1:
            a = [1, 2]
        elif metal_range == 2:
            a = [2, 3]            
        for j in range(2):
            text1 = 'z = %0.3f' '\n' % metals_z[a[j]]
            text2 = '[Fe/H] = %0.2f' % metals_feh[a[j]]
            text = text1+text2
            plt.plot(zam_met[a[j]][3], zam_met[a[j]][2], c='k', ls=lin_st[j],
                     lw=2., label=text)    
                         
        # Add legend.
        ax2.legend(loc="upper right", markerscale=1.5, scatterpoints=2,
                   fontsize=20)
        
        fig.tight_layout()
        # Generate output file for each data file.
        plt.savefig(join(out_dir+'fitted_zams/'+'final_ZAMS_%s.png' % \
        metal_range), dpi=150)
        
    
        # Plot CMD for all sequences only once.
        if metal_range == 1:
            
            # figsize(x1, y1), GridSpec(y2, x2) -> To have square plots:
            # x1/x2 = y1/y2 = 2.5 
            fig = plt.figure(figsize=(40, 25)) # create the top-level container
            gs = gridspec.GridSpec(10, 16)  # create a GridSpec object
        
            ax1 = plt.subplot(gs[1:9, 1:8])    
            plt.ylim(9, -3)
            plt.xlim(-1., 3.)
            plt.xlabel(r'$(C-T_1)_o$', fontsize=28)
            plt.ylabel(r'$M_{T_1}$', fontsize=28)
            # Ticks.
            ax1.xaxis.set_major_locator(MultipleLocator(1.0))
            ax1.minorticks_on()
            ax1.tick_params(which='minor', length=8)
            ax1.tick_params(which='major', length=12)
            ax1.grid(b=True, which='both', color='gray', linestyle='--', lw=1)
            ax1.tick_params(axis='both', which='major', labelsize=26)
            
            # Store ages in array.
            ages, names, names_feh, feh_list = [], [], [], []
            for seq_param in final_zams_params:
                ages.append(seq_param[2])
                names.append(seq_param[0])
                names_feh.append(seq_param[0]+' ('+str(seq_param[3])+')')
                feh_list.append(seq_param[3])
            ages = np.array(ages)    
                
            # Create new interpolated list.
            final_zams_poli = []
            for indx, seq in enumerate(final_zams):
                # Obtain and plot fitting polinome.
                poli = np.polyfit(seq[1], seq[0], 3)
                y_pol = np.linspace(min(seq[1]), max(seq[1]), 50)
                p = np.poly1d(poli)
                x_pol = [p(i) for i in y_pol]
                final_zams_poli.append([x_pol, y_pol])
            
            # Sort all lists according to age.
            ages_s, names_s, names_feh_s, final_zams_poli_s = \
            map(list, zip(*sorted(zip(ages, names, names_feh, final_zams_poli),
                                  reverse=True)))
                              
            cmap = plt.get_cmap('rainbow')
            k = 0
            for (x, y), color, label in zip(final_zams_poli_s, ages_s,\
            names_feh_s):
                # Transform color value.
                m, h = 1./(max(ages_s) - min(ages_s)), \
                min(ages_s)/(min(ages_s) - max(ages_s))
                col_transf = m*color+h
                l, = plt.plot(x, y, label=label, color=cmap(col_transf), lw=2.)
                pos = [(x[-2]+x[-1])/2.+0.15, (y[-2]+y[-1])/2.]
                pos = [x[-1], y[-1]+0.1]
                plt.text(pos[0], pos[1], names_s[k], size=16, rotation=0,
                         color=l.get_color(), ha="center", va="center",\
                         bbox=dict(ec='1',fc='1', alpha=0.6))        
                k += 1                
                
            m = plt.cm.ScalarMappable(cmap=cmap)
            m.set_array(ages_s)
            cbar = plt.colorbar(m)
            cbar.set_label(r'Age (Gyr)', fontsize=26, labelpad=20)
            cbar.ax.tick_params(which='major', length=12, labelsize=24)
            
            # Add text box
            ax1.legend(loc="upper right", markerscale=1.5, scatterpoints=2,
                       fontsize=16)
            text = r'%0.2f $\leq$ [Fe/H] $\leq$ %0.2f' % (min(feh_list),\
            max(feh_list))
            plt.text(0.355, 0.975, text, transform=ax1.transAxes,
                     bbox=dict(facecolor='gray', alpha=0.1,
                               boxstyle='round,pad=0.4'), fontsize=24)              
            
            # Add legend.        
            ax1.legend(loc="upper right", markerscale=1.5, scatterpoints=2,
                       fontsize=16)
                    
        
            ax2 = plt.subplot(gs[1:9, 9:15])    
            plt.ylim(9, -3)
            plt.xlim(-1., 3.)
            plt.xlabel(r'$(C-T_1)_o$', fontsize=28)
            plt.ylabel(r'$M_{T_1}$', fontsize=28)
            # Ticks.
            ax2.xaxis.set_major_locator(MultipleLocator(1.0))
            ax2.minorticks_on()
            ax2.tick_params(which='minor', length=8)
            ax2.tick_params(which='major', length=12)
            ax2.grid(b=True, which='both', color='gray', linestyle='--', lw=1)
            ax2.tick_params(axis='both', which='major', labelsize=26)
                
            # Rearrange sequences into single list composed of two sub-lists:
            # the first one holds the colors and the second one the magnitudes.
            single_seq_list = [[i for v in r for i in v] for r in \
            zip(*final_zams_poli_s)]
            
            # Obtain and plot fitting polinome for all sequences.
            poli_order = [3]
            pol_col = ['r', 'b']
            for j, order in enumerate(poli_order):
                poli_zams = np.polyfit(single_seq_list[1], single_seq_list[0],
                                       order)
                y_pol = np.linspace(min(single_seq_list[1]),
                                    max(single_seq_list[1]), 50)
                p = np.poly1d(poli_zams)
                x_pol = [p(i) for i in y_pol]
                plt.plot(x_pol, y_pol, c=pol_col[j], lw=2.5,
                         label='ZAMS (%d)' % order)        
        
            # Plot ZAMS envelope.
            lin_st = ['--', '-']
            a = [1, 3]
            for j in range(2):
                text1 = 'z = %0.3f' '\n' % metals_z[a[j]]
                text2 = '[Fe/H] = %0.2f' % metals_feh[a[j]]
                text = text1+text2
                plt.plot(zam_met[a[j]][3], zam_met[a[j]][2], c='k', ls=lin_st[j],
                         lw=2., label=text) 
                         
            # Add legend.
            ax2.legend(loc="upper right", markerscale=1.5, scatterpoints=2,
                       fontsize=20)
        
            fig.tight_layout()
            # Generate output file for each data file.
            plt.savefig(join(out_dir+'fitted_zams/'+'final_ZAMS_ALL.png'),
                        dpi=150)

    else:
        print 'Skipped %d' % metal_range