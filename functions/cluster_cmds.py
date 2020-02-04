# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:13:30 2013

@author: gabriel
"""

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator


# Exponential fitted function.
def exp_func(x, a, b, c):
    '''Exponential function.
    '''
    return a * np.exp(b * x) + c
    

def make_cluster_cmds(sub_dir, cluster, col1_data, mag_data, stars_out_rjct,
                      stars_out, stars_in_rjct, stars_in, prob_memb_avrg,
                      popt_mag, popt_col1, cl_e_bv, cl_age, cl_feh, cl_dmod,
                      iso_moved, iso_intrsc,zams_iso, col1_min_int, col1_max_int, 
                      mag_min_int, mag_max_int, min_prob, fine_tune, x, y, kde,
                      manual_levels, col_intrsc, mag_intrsc, memb_above_lim,
                      zam_met, x_pol, y_pol, out_dir):


    # Used when plotting all stars inside cluster radius with their
    # probability values (third plot).
    m_p_m_temp = [[], [], []]
    for star in prob_memb_avrg:
        m_p_m_temp[0].append(star[6])
        m_p_m_temp[1].append(star[4])
        m_p_m_temp[2].append(star[8])
        
    
    # Plot all outputs        
    # figsize(x1, y1), GridSpec(y2, x2) --> To have square plots: x1/x2 = 
    # y1/y2 = 2.5 
    fig = plt.figure(figsize=(20, 25)) # create the top-level container
    gs1 = gridspec.GridSpec(10, 8)  # create a GridSpec object
    
    fig.suptitle('\n%s' % sub_dir+'/'+cluster, fontsize=30)
   

    # Define plot limits for the first three CMD diagrams.
    col1_min, col1_max = max(-0.9, min(col1_data)-0.2),\
                         min(3.9, max(col1_data)+0.2)
    mag_min, mag_max = max(mag_data)+0.5, min(mag_data)-0.5          
    
   
    # Field stars CMD (stars outside cluster's radius)
    ax1 = plt.subplot(gs1[1:5, 0:4])
    #Set plot limits
    plt.xlim(col1_min, col1_max)
    plt.ylim(mag_min, mag_max)
    #Set axis labels
    plt.xlabel(r'$C-T_1$', fontsize=26)
    plt.ylabel(r'$T_1$', fontsize=26)
    ax1.tick_params(axis='both', which='major', labelsize=24)
    # Set minor ticks
    ax1.minorticks_on()
    # Only draw units on axis (ie: 1, 2, 3)
    ax1.xaxis.set_major_locator(MultipleLocator(1.0))
    # Set grid
    ax1.grid(b=True, which='major', color='gray', linestyle='--', lw=1)
    plt.text(0.05, 0.93, r'$r > R_{cl}$', transform = ax1.transAxes, \
    bbox=dict(facecolor='white', alpha=0.5), fontsize=26)
    # Plot stars.
    stars_rjct_temp = [[], []]
    for star in stars_out_rjct:
        stars_rjct_temp[0].append(star[5])
        stars_rjct_temp[1].append(star[3])
    plt.scatter(stars_rjct_temp[0], stars_rjct_temp[1], marker='x', c='k', 
                s=60, lw=1.5, zorder=1)
    stars_acpt_temp = [[], []]
    for star in stars_out:
        stars_acpt_temp[0].append(star[5])
        stars_acpt_temp[1].append(star[3])
    sz_pt = 10 if (len(stars_out_rjct)+len(stars_out)) > 5000 else 20
    plt.scatter(stars_acpt_temp[0], stars_acpt_temp[1], marker='o', c='k', 
                s=sz_pt, zorder=2)


    # Cluster's stars CMD (stars inside cluster's radius)
    ax2 = plt.subplot(gs1[1:5, 4:8])
    #Set plot limits
    plt.xlim(col1_min, col1_max)
    plt.ylim(mag_min, mag_max)
    #Set axis labels
    plt.xlabel(r'$C-T_1$', fontsize=26)
    plt.ylabel(r'$T_1$', fontsize=26)
    ax2.tick_params(axis='both', which='major', labelsize=24)
    # Set minor ticks
    ax2.minorticks_on()
    # Only draw units on axis (ie: 1, 2, 3)
    ax2.xaxis.set_major_locator(MultipleLocator(1.0))
    # Set grid
    ax2.grid(b=True, which='major', color='gray', linestyle='--', lw=1)
    # Calculate total number of stars whitin cluster's radius.
    tot_stars = len(stars_in_rjct) + len(stars_in)
    plt.text(0.68, 0.93, r'$r \leq R_{cl}\,|\,N_{c}=%d$' % tot_stars, 
             transform = ax2.transAxes, 
             bbox=dict(facecolor='white', alpha=0.5), fontsize=26)
    # Plot stars.
    stars_rjct_temp = [[], []]
    for star in stars_in_rjct:
        stars_rjct_temp[0].append(star[5])
        stars_rjct_temp[1].append(star[3])
    plt.scatter(stars_rjct_temp[0], stars_rjct_temp[1], marker='x', c='k', 
                s=60, lw=1.5, zorder=1)
    stars_acpt_temp = [[], []]
    for star in stars_in:
        stars_acpt_temp[0].append(star[5])
        stars_acpt_temp[1].append(star[3])
    sz_pt = 10 if (len(stars_in_rjct)+len(stars_in)) > 1000 else 20
    plt.scatter(stars_acpt_temp[0], stars_acpt_temp[1], marker='o', c='k', 
                s=sz_pt, zorder=2)
                
                
    # Cluster's stars CMD of stars inside the cluster's radius.
    ax3 = plt.subplot(gs1[5:9, 0:4])
    #Set plot limits
    plt.xlim(col1_min, col1_max)
    plt.ylim(mag_min, mag_max)
    #Set axis labels
    plt.xlabel(r'$C-T_1$', fontsize=26)
    plt.ylabel(r'$T_1$', fontsize=26)
    ax3.tick_params(axis='both', which='major', labelsize=24)
    text = r'$r \leq R_{cl}\,|\,N_{c}=%d$' % len(prob_memb_avrg)
    plt.text(0.05, 0.93, text, transform = ax3.transAxes,
             bbox=dict(facecolor='white', alpha=0.5), fontsize=26)
    # Set minor ticks
    ax3.minorticks_on()
    ax3.xaxis.set_major_locator(MultipleLocator(1.0))
    ax3.grid(b=True, which='major', color='gray', linestyle='--', lw=1)
    # This reversed colormap means higher prob stars will look redder.
    cm = plt.cm.get_cmap('RdYlBu_r')
    # Max and min limits for colorbar.
    v_min, v_max = round(min(m_p_m_temp[2]),2),\
    round(max(m_p_m_temp[2]),2)
    # Invert list so stars with higher probs will be on top.
    m_p_m_temp_inv = [i[::-1] for i in m_p_m_temp]
    plt.scatter(m_p_m_temp_inv[0], m_p_m_temp_inv[1], marker='o', 
                c=m_p_m_temp_inv[2], s=100, cmap=cm, lw=0.8, \
                vmin=v_min, vmax=v_max, zorder=5)
    # Plot error bars at several mag values.
    mag_y = np.arange(int(min(m_p_m_temp_inv[1])+0.5), 
                      int(max(m_p_m_temp_inv[1])+0.5) + 0.1)
    x_val = [min(3.9, max(col1_data)+0.2) - 0.4]*len(mag_y)
    plt.errorbar(x_val, mag_y, yerr=exp_func(mag_y, *popt_mag), 
                 xerr=exp_func(mag_y, *popt_col1), fmt='k.', lw=1.2, \
                 ms=0., zorder=4)
    # Plot ZAMS.
    plt.plot(zams_iso[1], zams_iso[0], c='k', ls='--', lw=1.5)
    # Plot moved isochrone.
    plt.plot(iso_moved[1], iso_moved[0], 'k', lw=2.)
    # Plot colorbar.
    cbaxes3 = fig.add_axes([0.4, 0.46, 0.07, 0.01])
    cbar3 = plt.colorbar(cax=cbaxes3, ticks=[v_min,v_max],
                         orientation='horizontal')
    cbar3.ax.tick_params(labelsize=15)
               

    # Plot the intrinsic CMD diagram along with the contour levels
    # and the ZAMS for each cluster.
    ax4 = plt.subplot(gs1[5:9, 4:8])
    #Set plot limits
    plt.xlim(col1_min_int, col1_max_int)
    plt.ylim(mag_min_int, mag_max_int)
    #Set axis labels
    plt.xlabel(r'$(C-T_1)_o$', fontsize=26)
    plt.ylabel(r'$M_{T_1}$', fontsize=26)
    ax4.tick_params(axis='both', which='major', labelsize=24)
    # Add text box
    text1 = r'$E_{(B-V)} = %0.2f}$' '\n' % cl_e_bv
    text2 = r'$Age = %0.3f}$' '\n' % cl_age
    text3 = r'$[Fe/H] = %0.2f}$' '\n' % cl_feh
    text4 = r'$(m-M)_o = %0.2f}$' % cl_dmod
    text = text1+text2+text3+text4
    plt.text(0.7, 0.83, text, transform = ax4.transAxes,
             bbox=dict(facecolor='white', alpha=0.5), fontsize=24)
    plt.text(0.05, 0.93, r'$P_{lim}=%0.2f$' % min_prob,
             transform=ax4.transAxes,
             bbox=dict(facecolor='white', alpha=0.5), fontsize=24)
    # Set minor ticks
    ax4.minorticks_on()
    ax4.xaxis.set_major_locator(MultipleLocator(1.0))
    ax4.yaxis.set_major_locator(MultipleLocator(1.0))
    ax4.grid(b=True, which='major', color='gray', linestyle='--', lw=1)
    # This reversed colormap means higher prob stars will look redder.
    cm = plt.cm.get_cmap('RdYlBu_r')
    # Plor contour levels.
    if fine_tune == True and manual_levels.any():
        CS = plt.contour(x, y, kde, manual_levels)
    else:
        CS = plt.contour(x, y, kde)
    plt.clabel(CS, fontsize=11, inline=1)
    # Invert list so stars with higher probs will be on top.
    temp_list = [col_intrsc, mag_intrsc, memb_above_lim[2]]
    temp_inv = [i[::-1] for i in temp_list]
    # Plot colored stars.
    plt.scatter(temp_inv[0], temp_inv[1], marker='o', c=temp_inv[2],
                s=100, cmap=cm, lw=0.8, vmin=v_min, vmax=v_max, zorder=2)
    # Plot ZAMS envelope.
    a = [0, -1]
    for j in a:
        plt.plot(zam_met[j][3], zam_met[j][2], c='g', ls='--') 
    # Plot polynomial fit only if list is not empty.
    if x_pol:
        plt.plot(x_pol, y_pol, c='k', lw=2, zorder=6)
    # Plot intrinsic isochrone.
    plt.plot(iso_intrsc[1], iso_intrsc[0], 'k', lw=1.5, ls='--')
                 

 
    fig.tight_layout()
    # Generate output file for each data file.
    plt.savefig(join(out_dir+'fitted_zams'+'/'+cluster+'_ZAMS.png'), dpi=150)
    # Close to release memory.
    plt.clf()
    plt.close()