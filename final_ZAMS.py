# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:49:53 2013

@author: gabriel
"""

'''
Script to produce a reduced image for each cluster. It only prints the full
finding chart for the frame, the field and cluster CMDs and the final
cluster CMD composed of only the most probable members.
It also outputs a data file for each cluster containing the ZAMS traced by its
members.


1- Read data_output file to store names and parameters of each cluster:
   sub dir, name, center, radius, number of members.
2- Read clusters_data_isos.dat file to store isochrone parameters for each
   cluster.
3- Read the photometric data file for each cluster.
4- Read most_prob_memb file for each cluster to store the probabilities
and CMD coordinates assigned to each star.

5- Create the finding chart using the x,y coordinates, the assigned center
coordinates and the radius.
6- Create the r>R_c CMD using the color and magnitude of each star, the
assigned center and radius for the cluster.
7- Create the cluster CMD using the same process as above.
8- Create the last CMD using the data from the last file read.
'''


import functions.get_data as gd
import functions.err_accpt_rejct as ear
import functions.get_in_out as gio
from functions.get_isochrones import get_isochrones as g_i

import numpy as np
from os import getcwd
from os.path import join, getsize, realpath, dirname
from os.path import expanduser
import glob
from scipy import stats
import scipy as sp
from itertools import chain

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator



# This list stores the clusters accepted by hand by Piatti.
piatti_accept = ['AM3', 'BSDL3158', 'BSDL594', 'BSDL761', 'C11', 'CZ26',
                 'CZ30', 'H88-26', 'H88-52', 'H88-55', 'HAF11', 'HS130',
                 'HS154', 'HS156', 'HS178', 'HS38', 'HW22', 'HW31', 'HW40',
                 'HW41', 'HW59', 'HW84', 'HW86', 'IC2146', 'KMHK1055',
                 'KMHK112', 'KMHK123', 'KMHK1668', 'KMHK1702', 'L102',
                 'L106', 'L108', 'L110', 'L111', 'L113', 'L114', 'L115',
                 'L30', 'L35', 'L45', 'L50', 'L58', 'L62', 'L72', 'LW211',
                 'LW224', 'LW263', 'LW469', 'LW69', 'NGC1644', 'NGC1697',
                 'NGC1838', 'NGC1839', 'NGC1863', 'NGC2093', 'NGC2108',
                 'NGC2236', 'NGC2324', 'NGC2627', 'NGC339', 'SL13', 'SL133',
                 'SL154', 'SL162', 'SL229', 'SL293', 'SL300', 'SL351', 'SL41',
                 'SL444', 'SL446A', 'SL5', 'SL510', 'SL54', 'SL588', 'SL663',
                 'SL674', 'SL707', 'SL72', 'SL73', 'SL869', 'SL870', 'SL96',
                 'TO1', 'TR5']

gabriel_accept = ['CZ26', 'CZ30', 'H88-55', 'H88-188', 'H88-245', 'H88-333', 
                  'HAF11', 'HS38', 'HS130', 'HS154', 'HS178', 'KMHK123',
                  'KMHK128', 'KMHK1702', 'L102', 'L114', 'L115', 'LW469',
                  'NGC2236', 'NGC2324', 'RUP1', 'SL72', 'SL154', 'TO1']
                  
ruben_accept = ['SL444', 'NGC1838', 'NGC1863', 'SL218', 'NGC1997', 'L35',
                'L45', 'L50', 'L62', 'L113', 'L111', 'L114', 'L115', 'L72',
                'CZ26', 'HAF11', 'NGC2236', 'NGC2324', 'RUP1', 'BSDL1035',
                'H88-26', 'H88-55', 'H88-333', 'HS38', 'LW469', 'NGC2093',
                'SL154', 'SL300', 'SL588', 'L58', 'NGC419', 'IC2146',
                'NGC1644', 'NGC2108', 'SL869', 'BSDL654', 'BSDL779', 'HS130']

def intrsc_values(col_obsrv, mag_obsrv, e_bv, dist_mod):
    '''
    Takes *observed* color and magnitude lists and returns corrected or
    intrinsic lists. Depends on the system selected/used.
    '''
    # For Washington system.
    #
    # E(C-T1) = 1.97*E(B-V) = (C-T1) - (C-T)o
    # M_T1 = T1 + 0.58*E(B-V) - (m-M)o - 3.2*E(B-V)
    #
    # (C-T1)o = (C-T1) - 1.97*E(B-V)
    # M_T1 = T1 + 0.58*E(B-V) - (m-M)o - 3.2*E(B-V)
    #
    col_intrsc = np.array(col_obsrv) - 1.97*e_bv
    mag_intrsc = np.array(mag_obsrv) + 0.58*e_bv - dist_mod - 3.2*e_bv
    
    return col_intrsc, mag_intrsc


# Exponential fitted function.
def func(x, a, b, c):
    '''Exponential function.
    '''
    return a * np.exp(b * x) + c
    



# This list holds the names and tuning parameters for those clusters that need
# it.
# 1st sub-list: Names of clusters
# 2nd: values to generate levels with np.arange()
# 3rd: x_min, x_max, y_min and y_max. Rabges to reject points.
# 4th: min level value to accept and min level number to accept.
f_t_names = ['H88-245', 'HS130', 'KMHK1702', 'LW469', 'RUP1', 'KMHK128', 'L115']

f_t_range = [[0., 1.01, 0.2], [0., 1.01, 0.05], [0.3, 2.11, 0.3], \
[0.2, 1.21, 0.2], [0.05, 0.351, 0.05], [0.05, 1.251, 0.05], [0.15, 0.91, 0.15]]

f_t_xylim = [[-1., 1., -0.2, 1.7], [-1., 1., 0.4, 3.6], [-1., 1., 1.3, 3.6],\
[-1., 1., 1.2, 4.], [-1., 3., 1., 6.8], [-1., 0.5, 1.8, 3.6], \
[-1., 1, -0.4, 3]]

f_t_level = [[0.1, 1], [-0.1, -1], [-0.1, -1], [-0.1, -1], [0.1, 1], [0.1, 1], \
[-0.1, -1]]

fine_tune_list = [f_t_names, f_t_range, f_t_xylim, f_t_level]

                  
def contour_levels(cluster, x, y, kde):
    '''This is the central function. It generates the countour plots around the
    cluster members. The points of these contour levels are used to trace the
    ZAMS for each cluster.
    '''

    fine_tune = False
    if cluster in fine_tune_list[0]:
        fine_tune = True
        indx = fine_tune_list[0].index(cluster)
        manual_levels = np.arange(fine_tune_list[1][indx][0],
                                  fine_tune_list[1][indx][1],\
                                  fine_tune_list[1][indx][2])
        x_min, x_max = fine_tune_list[2][indx][0], fine_tune_list[2][indx][1]
        y_min, y_max = fine_tune_list[2][indx][2], fine_tune_list[2][indx][3]
        lev_min, lev_num = fine_tune_list[3][indx]
    else:
        x_min, x_max = -10., 10.
        y_min, y_max = -10., 10.
        lev_min, lev_num = 0.1, 1

    # This list will hold the points obtained through the contour curves,
    # the first sublist are the x coordinates of the points and the second
    # the y coordinates.
    sequence = [[], []]

    # Store contour levels.
    if fine_tune == True:
        CS = plt.contour(x, y, kde, manual_levels)
    else:
        CS = plt.contour(x, y, kde)
    plt.clabel(CS, fontsize=9, inline=1)
    # Store level values for contour levels.
    levels = CS.levels
#    print levels
    for i,clc in enumerate(CS.collections):
        for j,pth in enumerate(clc.get_paths()):
            cts = pth.vertices
            d = sp.spatial.distance.cdist(cts,cts)
            x_c,y_c = cts[list(sp.unravel_index(sp.argmax(d),d.shape))].T
            # Only store points that belong to contour PDF values larger
            # tnah lev_min and that belong to the uper curves, ie: do not
            # use those with index <= lev_num.
            if levels[i] > lev_min and i > lev_num:
                # Only store points within these limits.
                if x_min <= x_c[0] <= x_max and y_min <= y_c[0] <= y_max:
                    sequence[0].append(round(x_c[0],4))
                    sequence[1].append(round(y_c[0],4))
                if x_min <= x_c[1] <= x_max and y_min <= y_c[1] <= y_max:    
                    sequence[0].append(round(x_c[1],4))
                    sequence[1].append(round(y_c[1],4))

    return sequence
            

# *********************************

# Set 'home' dir.
home = expanduser("~")

# 1- Read data_output file to store names and parameters of each cluster:
# sub dir, name, center, radius, number of members.

# Location of the data_output file
out_dir = home+'/clusters_out/washington_KDE-Scott/'
data_out_file = out_dir+'data_output'

sub_dirs, cl_names, centers, radius, members = [], [], [[],[]], [], []
with open(data_out_file, mode="r") as d_o_f:
    for line in d_o_f:
        li=line.strip()
        # Jump comments.
        if not li.startswith("#"):
            reader = li.split()            
            sub_dirs.append(reader[0].split('/')[0])
            cl_names.append(reader[0].split('/')[1])
            centers[0].append(float(reader[1]))
            centers[1].append(float(reader[2]))
            radius.append(float(reader[3]))
            members.append(float(reader[7]))


# *********************************
# 2- Read clusters_data_isos.dat' file to store isochrone parameters for each
# cluster.

# Location of the data_input file
main_dir = '/media/rest/Dropbox/GABRIEL/CARRERA/3-POS-DOC/trabajo/'
data_isos_file = main_dir+'codigo/clusters_data_isos.dat'

extin, ages, metal, dist_mods = [], [], [], []
for cluster in cl_names:
    with open(data_isos_file, mode="r") as d_i_f:
        for line in d_i_f:
            li=line.strip()
            # Jump comments.
            if not li.startswith("#"):
                reader = li.split()     
                if reader[0] == cluster:
                    extin.append(reader[1])
                    ages.append(float(reader[2]))
                    metal.append(float(reader[3]))
                    dist_mods.append(float(reader[4]))
                    break


# *********************************
# 3- Read the photometric data file for each cluster.

# Location of the photometric data file for each cluster.
data_phot = main_dir+'data_all/cumulos-datos-fotometricos/'


# Stores the CMD sequence obtained for each cluster.
final_zams = []
# Also store the parameters associated with this cluster.
final_zams_params = []

for indx, sub_dir in enumerate(sub_dirs):
    cluster = cl_names[indx]
    
#    if cluster in gabriel_accept:
    if cluster in ruben_accept:
#    if cluster == 'L115':
#    use_all_clusters = True
#    if use_all_clusters:
        print sub_dir, cluster
        
        filename = glob.glob(join(data_phot, sub_dir, cluster + '.*'))[0]
        id_star, x_data, y_data, mag_data, e_mag, col1_data, e_col1 = \
        gd.get_data(data_phot, sub_dir, filename)
        
        # Accept and reject stars based on their errors.
        bright_end, popt_mag, popt_umag, pol_mag, popt_col1, popt_ucol1, \
        pol_col1, mag_val_left, mag_val_right, col1_val_left, col1_val_right, \
        acpt_stars, rjct_stars = ear.err_accpt_rejct(id_star, x_data, y_data,
                                                     mag_data, e_mag, col1_data,
                                                     e_col1)

        clust_rad = [radius[indx], 0.]
        center_cl = [centers[0][indx], centers[1][indx]]
        # Get stars in and out of cluster's radius.
        stars_in, stars_out, stars_in_rjct, stars_out_rjct =  \
        gio.get_in_out(center_cl, clust_rad[0], acpt_stars, rjct_stars)
        
        # Path where the code is running
        mypath = realpath(join(getcwd(), dirname(__file__)))
        clust_name = cluster
        # Get manually fitted parameters for cluster, if these exist.
        cl_e_bv, cl_age, cl_feh, cl_dmod, iso_moved, zams_iso = g_i(mypath,
                                                                    clust_name)
                                                     

    #4- Read most_prob_memb file for each cluster to store the probabilities
    # and CMD coordinates assigned to each star.
        most_prob_memb_avrg = []
        file_path = join(out_dir+sub_dir+'/'+cluster+'_memb.dat')
        with open(file_path, mode="r") as m_f:
            # Check if file is empty.
            flag_area_stronger = False if getsize(file_path) > 44 else True
            for line in m_f:
                li=line.strip()
                # Jump comments.
                if not li.startswith("#"):
                    reader = li.split()     
                    if reader[0] == '99.0':
                        most_prob_memb_avrg.append(map(float, reader))
    
    
        # Plot all outputs
        
        # Define plot limits for ALL CMD diagrams.
        col1_min, col1_max = max(-0.9, min(col1_data)-0.2),\
                             min(3.9, max(col1_data)+0.2)
        mag_min, mag_max = max(mag_data)+0.5, min(mag_data)-0.5    
        
        # figsize(x1, y1), GridSpec(y2, x2) --> To have square plots: x1/x2 = 
        # y1/y2 = 2.5 
        fig = plt.figure(figsize=(20, 25)) # create the top-level container
        gs1 = gridspec.GridSpec(10, 8)  # create a GridSpec object
#        fig = plt.figure(figsize=(25, 20)) # create the top-level container
#        gs1 = gridspec.GridSpec(8, 10)  # create a GridSpec object
        
        fig.suptitle('\n%s' % sub_dir+'/'+cluster, fontsize=30)
        
        # Field stars CMD (stars outside cluster's radius)
        ax1 = plt.subplot(gs1[1:5, 0:4])
#        ax1 = plt.subplot(gs1[1:7, 0:5])
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
#        ax2 = plt.subplot(gs1[1:7, 5:10])
        #Set plot limits
        plt.xlim(col1_min, col1_max)
        plt.ylim(mag_min, mag_max)
#        plt.xlim(-2., 4.)
#        plt.ylim(10., -4)
        #Set axis labels
        plt.xlabel(r'$C-T_1$', fontsize=26)
        plt.ylabel(r'$T_1$', fontsize=26)
#        plt.xlabel(r'$(C-T_1)_o$', fontsize=26)
#        plt.ylabel(r'$M_{T_1}$', fontsize=26)
        ax2.tick_params(axis='both', which='major', labelsize=24)
        # Set minor ticks
        ax2.minorticks_on()
        # Only draw units on axis (ie: 1, 2, 3)
        ax2.xaxis.set_major_locator(MultipleLocator(1.0))
        # Set grid
        ax2.grid(b=True, which='major', color='gray', linestyle='--', lw=1)
        # Calculate total number of stars whitin cluster's radius.
        tot_stars = len(stars_in_rjct) + len(stars_in)
        plt.text(0.68, 0.93, r'$r \leq R_{cl}\,|\,N=%d$' % tot_stars, 
                 transform = ax2.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.5), fontsize=26)
        # Plot stars.
        stars_rjct_temp = [[], []]
        for star in stars_in_rjct:
            stars_rjct_temp[0].append(star[5])
            stars_rjct_temp[1].append(star[3])
        # Get intrinsic color and magnitudes.
#        col_intrsc_r, mag_intrsc_r = intrsc_values(stars_rjct_temp[0],
#                                               stars_rjct_temp[1],\
#                                               cl_e_bv, cl_dmod)             
        plt.scatter(stars_rjct_temp[0], stars_rjct_temp[1], marker='x', c='k', 
                    s=60, lw=1.5, zorder=1)
#        plt.scatter(col_intrsc_r, mag_intrsc_r, marker='x', c='k', 
#                    s=60, lw=1.5, zorder=1)
        stars_acpt_temp = [[], []]
        for star in stars_in:
            stars_acpt_temp[0].append(star[5])
            stars_acpt_temp[1].append(star[3])
        # Get intrinsic color and magnitudes.
        col_intrsc_a, mag_intrsc_a = intrsc_values(stars_acpt_temp[0],
                                               stars_acpt_temp[1],\
                                               cl_e_bv, cl_dmod)  
        sz_pt = 10 if (len(stars_in_rjct)+len(stars_in)) > 1000 else 20
        plt.scatter(col_intrsc_a, mag_intrsc_a, marker='o', c='k', 
                    s=sz_pt, zorder=2)
        # Add text box
        text1 = r'$E_{(B-V)} = %0.2f}$' '\n' % cl_e_bv
        text2 = r'$Age = %0.3f}$' '\n' % cl_age
        text3 = r'$[Fe/H] = %0.2f}$' '\n' % cl_feh
        text4 = r'$(m-M)_o = %0.2f}$' % cl_dmod
        text = text1+text2+text3+text4
        plt.text(0.75, 0.02, text, transform = ax2.transAxes,
                 bbox=dict(facecolor='white', alpha=0.5), fontsize=24)
                    
                    
        # Cluster's stars CMD of N_c stars (the approx number of member stars)
        # inside cluster's radius with the smallest decontamination index 
        # (most probable members).
        # Check if decont algorithm was applied.
        if not(flag_area_stronger):
            ax3 = plt.subplot(gs1[5:9, 0:4])
            #Set plot limits
            plt.xlim(col1_min, col1_max)
            plt.ylim(mag_min, mag_max)
            #Set axis labels
            plt.xlabel(r'$C-T_1$', fontsize=26)
            plt.ylabel(r'$T_1$', fontsize=26)
            ax3.tick_params(axis='both', which='major', labelsize=24)
            text = r'$r \leq R_{cl}\,|\,N_{c}=%d$' % len(most_prob_memb_avrg)
            plt.text(0.05, 0.93, text, transform = ax3.transAxes,
                     bbox=dict(facecolor='white', alpha=0.5), fontsize=26)
            # Set minor ticks
            ax3.minorticks_on()
            ax3.xaxis.set_major_locator(MultipleLocator(1.0))
            ax3.grid(b=True, which='major', color='gray', linestyle='--', lw=1)
            # This reversed colormap means higher prob stars will look redder.
            cm = plt.cm.get_cmap('RdYlBu_r')
            m_p_m_temp = [[], [], []]
            for star in most_prob_memb_avrg:
                m_p_m_temp[0].append(star[6])
                m_p_m_temp[1].append(star[4])
                m_p_m_temp[2].append(star[8])
            # Create new list with inverted values so higher prob stars are on top.
            m_p_m_temp_inv = [i[::-1] for i in m_p_m_temp]
            plt.scatter(m_p_m_temp_inv[0], m_p_m_temp_inv[1], marker='o', 
                        c=m_p_m_temp_inv[2], s=100, cmap=cm, lw=0.8, vmin=0, vmax=1,\
                        zorder=5)
            # If list is not empty.
            if m_p_m_temp_inv[1]:
                # Plot error bars at several mag values.
                mag_y = np.arange(int(min(m_p_m_temp_inv[1])+0.5), 
                                  int(max(m_p_m_temp_inv[1])+0.5) + 0.1)
                x_val = [min(3.9, max(col1_data)+0.2) - 0.4]*len(mag_y)
                plt.errorbar(x_val, mag_y, yerr=func(mag_y, *popt_mag), 
                             xerr=func(mag_y, *popt_col1), fmt='k.', lw=1.2, ms=0.,\
                             zorder=4)
            # Plot ZAMS.
            plt.plot(zams_iso[1], zams_iso[0], c='k', ls='--', lw=1.5)
            # Plot isochrone.
            plt.plot(iso_moved[1], iso_moved[0], 'k', lw=2.)
                   
    

        # Now we plot the intrinsic CMD diagram along with the contour levels
        # and the ZAMS for each cluster.
        
        # Get intrinsic color and magnitudes. Use inverted lists so the values
        # that return are already ordered according to theri weights.
        col_intrsc, mag_intrsc = intrsc_values(m_p_m_temp_inv[0],
                                               m_p_m_temp_inv[1],\
                                               cl_e_bv, cl_dmod) 
                          
                          
        # Obtain new limits selected as to make the intrinsic CMD axis 1:1.
        col1_min, col1_max = min(col_intrsc)-0.2, max(col_intrsc)+0.2
        mag_min, mag_max = max(mag_intrsc)+1., min(mag_intrsc)-1.
        delta_x = col1_max - col1_min
        delta_y = mag_min - mag_max
        center_x = (col1_max + col1_min)/2.
        center_y = (mag_max + mag_min)/2.
        if delta_y >= delta_x:
            col1_min, col1_max = (center_x-delta_y/2.), (center_x+delta_y/2.)
        else:
            mag_max, mag_min = (center_y-delta_x/2.), (center_y+delta_x/2.)
        
        
        # Generate new stars located at the same positions of each star in the list
        # of most probable members. The number of new stars generated in each star
        # position is the weight assigned to that star times 10. We do this so
        # the KDE obtained below incorporates the information of the weights, ie:
        # the membership probabilities.
        col_intrsc_w = list(chain.from_iterable([i] * int(round(j* 10)) \
        for i, j in zip(col_intrsc, m_p_m_temp_inv[2])))
        mag_intrsc_w = list(chain.from_iterable([i] * int(round(j* 10)) \
        for i, j in zip(mag_intrsc, m_p_m_temp_inv[2])))
    
    
                    
        if not(flag_area_stronger):
            ax4 = plt.subplot(gs1[5:9, 4:8])
            #Set plot limits
            plt.xlim(col1_min, col1_max)
            plt.ylim(mag_min, mag_max)
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
            # Set minor ticks
            ax4.minorticks_on()
            ax4.xaxis.set_major_locator(MultipleLocator(1.0))
            ax4.yaxis.set_major_locator(MultipleLocator(1.0))
            ax4.grid(b=True, which='major', color='gray', linestyle='--', lw=1)
            # This reversed colormap means higher prob stars will look redder.
            cm = plt.cm.get_cmap('RdYlBu_r')
            plt.scatter(col_intrsc, mag_intrsc, marker='o', c=m_p_m_temp_inv[2],
                        s=100, cmap=cm, lw=0.8, vmin=0, vmax=1, zorder=2)
    
            # Get KDE for CMD intrinsic position of most probable members.
            x, y = np.mgrid[col1_min:col1_max:100j, mag_min:mag_max:100j]
            positions = np.vstack([x.ravel(), y.ravel()])
            values = np.vstack([col_intrsc_w, mag_intrsc_w])
            # The results are HEAVILY dependant on the bandwidth used here.
            # See: http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
            kernel = stats.gaussian_kde(values, bw_method = None)
            kde = np.reshape(kernel(positions).T, x.shape)
            
            # Call the function that returns the sequence determined by the two
            # points further from each other in each contour level.
            sequence = contour_levels(cluster, x, y, kde)
    
            # Obtain and plot the sequence's fitting polinome.
            poli_order = 3 # Order of the polynome.
            poli = np.polyfit(sequence[1], sequence[0], poli_order)
            y_pol = np.linspace(min(sequence[1]), max(sequence[1]), 50)
            p = np.poly1d(poli)
            x_pol = [p(i) for i in y_pol]
            plt.plot(x_pol, y_pol, c='k', lw=2, zorder=6)
            
            # Write data to output file.
            out_file = join(out_dir+'fitted_zams'+'/'+cluster+'_ZAMS.dat')
            name = [sub_dir+'/'+cluster]*len(x_pol)
            e_bv = [str(cl_e_bv)]*len(x_pol)
            age = [str(cl_age)]*len(x_pol)
            feh = [str(cl_feh)]*len(x_pol)
            dmod = [str(cl_dmod)]*len(x_pol)
            line = zip(*[name, ['%.2f' % i for i in x_pol],
                         ['%.2f' % i for i in y_pol], e_bv, age, feh, dmod])
            with open(out_file, 'w') as f_out:
                f_out.write("#Name x_zams y_zams E(B-V) Age (Gyr) [Fe/H] (m-M)o\n")
                for item in line:
                    f_out.write('{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(*item))
                    f_out.write('\n')
    
     
        fig.tight_layout()
        # Generate output file for each data file.
        plt.savefig(join(out_dir+'fitted_zams'+'/'+cluster+'_ZAMS.png'), dpi=150)
#        plt.savefig(join(out_dir+'fitted_zams/ruben'+'/'+cluster+'_ruben.png'), dpi=150)
        # Close to release memory.
        plt.clf()
        plt.close()

        
        # Store the sequence obtained with this cluster in final list.
        final_zams.append(sequence)
        # Also store the parameters associated with this cluster.
        final_zams_params.append([cluster, cl_e_bv, cl_age, cl_feh, cl_dmod])



#print 'stop'
#raw_input()

# Second part of the code.

# Get ZAMS.
zams_file = 'codigo/zams_4_isos.data'
data = np.loadtxt(zams_file, unpack=True)
# Convert z to [Fe/H] using the y=A+B*log10(x) zunzun.com function and the
# x,y values:
#   z    [Fe/H]
# 0.001  -1.3
# 0.004  -0.7
# 0.008  -0.4
# 0.019  0.0
#    A, B = 1.7354259305164, 1.013629121876
#    feh = A + B*np.log10(z)
metals_z = [0.001, 0.004, 0.008, 0.019]
metals_feh = [-1.3, -0.7, -0.4, 0.0]

# List that holds all the isochroned of different metallicities.
zam_met = [[] for _ in range(len(metals_z))]

# Store each isochrone of a given metallicity in a list.
for indx, metal_val in enumerate(metals_z):
    zam_met[indx] = map(list, zip(*(col for col in zip(*data) if\
    col[0] == metal_val)))
    
        
    
    
def make_final_plot(metal_range):
    # Print the final plot.
    
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
        
        # figsize(x1, y1), GridSpec(y2, x2) -> To have square plots: x1/x2 = y1/y2 = 2.5 
        fig = plt.figure(figsize=(40, 25)) # create the top-level container
        gs = gridspec.GridSpec(10, 16)  # create a GridSpec object
    #    fig, ax1 = plt.subplots()
    
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
            m, h = 1./(max(ages_s) - min(ages_s)), min(ages_s)/(min(ages_s) - max(ages_s))
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
        ax1.legend(loc="upper right", markerscale=1.5, scatterpoints=2, fontsize=16)
        # Add text box
        text = r'%0.2f $\leq$ [Fe/H] $\leq$ %0.2f' % (metal_min, metal_max )
        plt.text(0.355, 0.975, text, transform=ax1.transAxes,
                 bbox=dict(facecolor='gray', alpha=0.1, boxstyle='round,pad=0.4'),\
                 fontsize=24)
                 
                
                
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
            
        # Rearrange sequences into single list composed of two sub-lists: the first
        # one holds the colors and the second one the magnitudes.
        single_seq_list = [[i for v in r for i in v] for r in zip(*final_zams_poli_s)]
        
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
        ax2.legend(loc="upper right", markerscale=1.5, scatterpoints=2, fontsize=20)
        
        fig.tight_layout()
        # Generate output file for each data file.
        plt.savefig(join(out_dir+'final_ZAMS_%s.png' % metal_range), dpi=150)
        
    
        # Plot CMD for all sequences only once.
        if metal_range == 1:
            
            # figsize(x1, y1), GridSpec(y2, x2) -> To have square plots: x1/x2 = y1/y2 = 2.5 
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
            for (x, y), color, label in zip(final_zams_poli_s, ages_s, names_feh_s):
                # Transform color value.
                m, h = 1./(max(ages_s) - min(ages_s)), min(ages_s)/(min(ages_s) - max(ages_s))
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
            text = r'%0.2f $\leq$ [Fe/H] $\leq$ %0.2f' % (min(feh_list), max(feh_list))
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
                
            # Rearrange sequences into single list composed of two sub-lists: the first
            # one holds the colors and the second one the magnitudes.
            single_seq_list = [[i for v in r for i in v] for r in zip(*final_zams_poli_s)]
            
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
            plt.savefig(join(out_dir+'final_ZAMS_ALL.png'), dpi=150)        

    else:
        print 'Skipped %d' % metal_range

# Call for the 3 metallicty ranges.    
for i in range(3):
    print 'Plotting %d' % i
    make_final_plot(i)


print 'End.'