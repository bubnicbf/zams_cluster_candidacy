# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:49:53 2013

@author: gabriel
"""

'''
Trace a cluster's star sequence using the intrinsic (corrected) CMD and the
probabilities assigned to each star by a decontamination algorithm.

See README.md file for more information.
'''


import functions.get_data as gd
import functions.err_accpt_rejct as ear
import functions.get_in_out as gio
from functions.get_isochrones import get_isochrones as g_i
from functions.cluster_cmds import make_cluster_cmds as m_c_c
from functions.final_plot import make_final_plot as m_f_p

import numpy as np
from os.path import join, getsize
from os.path import expanduser
import glob
from scipy import stats
import scipy as sp
from itertools import chain

import matplotlib.pyplot as plt
from scipy.stats import norm



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


# This list store the clusters accepted manually.
manual_accept = ['BSDL654', 'BSDL761', 'BSDL779', 'C11', 'CZ26', 'CZ30',
                   'H88-188', 'H88-333', 'HAF11', 'HS38', 'HS130',
                   'KMHK1702', 'L49', 'L50', 'L114', 'LW469', 'NGC2236',
                   'NGC2324', 'RUP1', 'SL72', 'TO1']

#manual_accept = ['BSDL654', 'BSDL761', 'BSDL779']
                   

def contour_levels(fine_tune, cluster, x, y, kde):
    '''This is the central function. It generates the countour plots around the
    cluster members. The extreme points of these contour levels are used to trace
    the ZAMS fiducial line for each cluster.
    '''
    
    # This list holds the names and tuning parameters for those clusters that
    # need it.
    # 1st sub-list: Names of clusters
    # 2nd: values to generate levels with np.arange()
    # 3rd: y_min and y_max. Range where sequence will be interpolated.
    # 4th: min level value to accept and min level number to accept.
    f_t_names = [\
    'BSDL654', 'BSDL761', 'C11', 'CZ26', 'CZ30', 'HAF11', \
    'H88-188', 'H88-333', 'HS38', 'HS130', 'KMHK128', 'KMHK1702', \
    'L45', 'L49', 'L50', 'L114', 'LW469', 'NGC2236', \
    'NGC2324', 'RUP1', 'SL72', 'TO1']
    
    f_t_ylim = [\
    [0.5, 3.2], [1., 2.6], [1.4, 4.], [3.2, 5.2], [3., 4.8], [2.2, 5.], \
    [1.8, 2.7], [2., 4.], [2., 3.], [1., 3.4], [2.4, 3.6], [2., 4.], \
    [0., 1.2], [-0.3, 1.2], [0.8, 1.8], [1.6, 2.8], [2., 3.2], [2., 4.5], \
    [1.8, 5.], [2., 6.4], [1., 2.8], [2.8, 4.4]]
    
    f_t_level = [\
    [], [], [-0.1, 0.], [], [], [-0.1, 1], \
    [], [], [], [-0.1, 0.], [-0.1, 2], [-0.1, -1.], \
    [], [-0.1, 0.], [-0.1, 1], [], [-0.1, 0.], [-0.1, 1.], \
    [], [-0.1, 2.], [], []]
    
    f_t_range = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                 [], [], [], [], [], []]
    
    fine_tune_list = [f_t_names, f_t_range, f_t_ylim, f_t_level]

    if fine_tune == True and cluster in fine_tune_list[0]:
        indx = fine_tune_list[0].index(cluster)
        if fine_tune_list[1][indx]:
            manual_levels = np.arange(fine_tune_list[1][indx][0],
                                      fine_tune_list[1][indx][1],\
                                      fine_tune_list[1][indx][2])
        else:
            manual_levels = np.array([])
        y_min, y_max = fine_tune_list[2][indx][0], fine_tune_list[2][indx][1]
        lev_min, lev_num = fine_tune_list[3][indx] if fine_tune_list[3][indx] \
        else [-0.1, -1]
    else:
        manual_levels = np.array([])
        y_min, y_max = -10., 10.
        lev_min, lev_num = 0.1, 1

    # This list will hold the points obtained through the contour curves,
    # the first sublist are the x coordinates of the points and the second
    # the y coordinates.
    sequence = [[], []]

    # Store contour levels.
    if fine_tune == True and manual_levels.any():
        CS = plt.contour(x, y, kde, manual_levels)
    else:
        CS = plt.contour(x, y, kde)
    # Store level values for contour levels.
    levels = CS.levels
    for i,clc in enumerate(CS.collections):
        for j,pth in enumerate(clc.get_paths()):
            cts = pth.vertices
            d = sp.spatial.distance.cdist(cts,cts)
            x_c,y_c = cts[list(sp.unravel_index(sp.argmax(d),d.shape))].T
            # Only store points that belong to contour PDF values larger
            # than lev_min and that belong to the uper curves, ie: do not
            # use those with index <= lev_num.
            if levels[i] > lev_min and i > lev_num:
                # Only store points within these limits.
                    sequence[0].append(round(x_c[0],4))
                    sequence[1].append(round(y_c[0],4))
                    sequence[0].append(round(x_c[1],4))
                    sequence[1].append(round(y_c[1],4))

    # Range in y axis for accepting interpolated values for the sequence.
    y_lim = [y_min, y_max]
    return sequence, manual_levels, y_lim
            


def get_cluster_params():
    '''
    Read data_output file to store names and parameters of each cluster:
    sub dir, name, center and radius.
    '''
    
    # Set 'home' dir.
    home = expanduser("~")
    
    # Location of the data_output file
    out_dir = home+'/clusters/clusters_out/washington_KDE-Scott/'
    data_out_file = out_dir+'data_output'
    
    sub_dirs, cl_names, centers, radius = [], [], [[],[]], []
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
                
    return sub_dirs, cl_names, centers, radius, out_dir



def get_zams():
    '''
    Get ZAMS in its intrinsic position from file for plotting purposes.
    '''
    
    zams_file = 'zams.data'
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
    metals_z = [0.001, 0.004, 0.008, 0.0098, 0.0138, 0.019]
    metals_feh = [-1.3, -0.7, -0.4, -0.3, -0.15, 0.0]
    
    # List that holds all the isochrones of different metallicities.
    zam_met = [[] for _ in range(len(metals_z))]
    
    # Store each isochrone of a given metallicity in a list.
    for indx, metal_val in enumerate(metals_z):
        zam_met[indx] = map(list, zip(*(col for col in zip(*data) if\
        col[0] == metal_val)))
    
    return zam_met, metals_z, metals_feh



def get_probs(out_dir, sub_dir, cluster):
    '''
    Read the members file for each cluster and store the probabilities
    and CMD coordinates assigned to each star in the cluster region.
    '''
    prob_memb_avrg = []
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
                    prob_memb_avrg.append(map(float, reader))
                    
    return flag_area_stronger, prob_memb_avrg
                        
                        

def write_seq_file(out_dir, cluster, x_pol, y_pol):
    '''
    Write interpolated sequence to output file.
    '''
    out_file = join(out_dir+'fitted_zams'+'/'+cluster+'_ZAMS.dat')
    line = zip(*[['%.2f' % i for i in x_pol], ['%.2f' % i for i in y_pol]])
    with open(out_file, 'w') as f_out:
        f_out.write("#x_zams y_zams\n")
        for item in line:
            f_out.write('{:<7} {:>5}'.format(*item))
            f_out.write('\n')


# **********************************************************************
# End of functions.



# Call function to obtain clusters locations, names. etc.
sub_dirs, cl_names, centers, radius, out_dir = get_cluster_params()


# Get ZAMS located at its instinsic position for plotting purposes.
zam_met, metals_z, metals_feh = get_zams()


# Ask for minimum probability threshold.
use_mu = False
prob_quest = raw_input('Use mu as probability threshold? (y/n): ')
if prob_quest == 'y':
    use_mu = True
else:
    min_prob = float(raw_input('Input minimum probability value to use: '))
    
    
# Ask if fine tuning parameters should be used upon contour levels.
tune_quest = raw_input('Use fine tuning parameters? (y/n): ')
if tune_quest == 'y':
    fine_tune = True
else:
    fine_tune = False


# Ask if all clusters should be processed or only those in a list.
clust_quest = raw_input('Use all clusters? (y/n): ')
if clust_quest == 'y':
    clust_list = cl_names
else:
    clust_list = manual_accept


# Stores the CMD sequence obtained for each cluster.
final_zams = []
# Also store the parameters associated with each cluster.
final_zams_params = []
# Loop through all clusters processed.
for indx, sub_dir in enumerate(sub_dirs):
    cluster = cl_names[indx]

    # Check if cluster is in list.    
    if cluster in clust_list:
        print sub_dir, cluster
        
        # Location of the photometric data file for each cluster.
        data_phot = '/media/rest/Dropbox/GABRIEL/CARRERA/3-POS-DOC/trabajo/\
data_all/cumulos-datos-fotometricos/'        
        
        # Get photometric data for cluster.
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
        ocaat_path = '/media/rest/github/OCAAT_code/'
        clust_name = cluster
        # Get manually fitted parameters for cluster, if these exist.
        cl_e_bv, cl_age, cl_feh, cl_dmod, iso_moved, zams_iso, iso_intrsc = \
        g_i(ocaat_path, clust_name)
                                                     
        # Read the members file for each cluster and store the probabilities
        # and CMD coordinates assigned to each star in the cluster region.
        flag_area_stronger, prob_memb_avrg = get_probs(out_dir, sub_dir, cluster)
    
    
        # Check if decont algorithm was applied.
        if not(flag_area_stronger):
            
            # Fit gaussian to probabilities distribution. The mean will act as
            # the prob threshold. Only stars with prob values above this mean
            # will be used to trace the sequence.
            if use_mu == True:
                prob_data = [star[8] for star in prob_memb_avrg]
                # Best Gaussian fit of data.
                (mu, sigma) = norm.fit(prob_data)
                min_prob = mu

            # Create list with stars with probs above min_prob.
            memb_above_lim = [[], [], []]
            for star in prob_memb_avrg:
                if star[8] >= min_prob:
                    memb_above_lim[0].append(star[6])
                    memb_above_lim[1].append(star[4])
                    memb_above_lim[2].append(star[8])

            # Get intrinsic color and magnitudes.
            col_intrsc, mag_intrsc = intrsc_values(memb_above_lim[0],
                                                   memb_above_lim[1], cl_e_bv, \
                                                   cl_dmod) 

            # Obtain limits selected as to make the intrinsic CMD axis 1:1.
            col1_min_int, col1_max_int = min(col_intrsc)-0.2, max(col_intrsc)+0.2
            mag_min_int, mag_max_int = max(mag_intrsc)+1., min(mag_intrsc)-1.
            delta_x = col1_max_int - col1_min_int
            delta_y = mag_min_int - mag_max_int
            center_x = (col1_max_int + col1_min_int)/2.
            center_y = (mag_max_int + mag_min_int)/2.
            if delta_y >= delta_x:
                col1_min_int, col1_max_int = (center_x-delta_y/2.),\
                (center_x+delta_y/2.)
            else:
                mag_max_int, mag_min_int = (center_y-delta_x/2.),\
                (center_y+delta_x/2.) 
                          
                          
            # Generate new stars located at the same positions of each star in
            # the list of most probable members. The number of new stars
            # generated in each star position is the weight assigned to that
            # star times 10. We do this so the KDE obtained below incorporates
            # the information of the weights, ie: the membership probabilities.
            col_intrsc_w = list(chain.from_iterable([i] * int(round(j* 10)) \
            for i, j in zip(col_intrsc, memb_above_lim[2])))
            mag_intrsc_w = list(chain.from_iterable([i] * int(round(j* 10)) \
            for i, j in zip(mag_intrsc, memb_above_lim[2])))        
        
  
            # Get KDE for CMD intrinsic position of most probable members.
            x, y = np.mgrid[col1_min_int:col1_max_int:100j,
                            mag_min_int:mag_max_int:100j]
            positions = np.vstack([x.ravel(), y.ravel()])
            values = np.vstack([col_intrsc_w, mag_intrsc_w])
            # The results are HEAVILY dependant on the bandwidth used here.
            # See: http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
            kernel = stats.gaussian_kde(values, bw_method = None)
            kde = np.reshape(kernel(positions).T, x.shape)
            
            # Call the function that returns the sequence determined by the two
            # points further from each other in each contour level.
            sequence, manual_levels, y_lim = contour_levels(fine_tune, cluster,
                                                            x, y, kde)
        
            # If the contour points returns an empty list don't attempt to
            # plot the polynomial fit.
            if sequence[0]:

                # Obtain the sequence's fitting polinome.
                poli_order = 3 # Order of the polynome.
                poli = np.polyfit(sequence[1], sequence[0], poli_order)
                y_pol = np.linspace(min(sequence[1]), max(sequence[1]), 50)
                p = np.poly1d(poli)
                x_pol = [p(i) for i in y_pol]

                # Trim the interpolated sequence to the range in y axis.
                y_pol_trim, x_pol_trim = zip(*[(ia,ib) for (ia, ib) in \
                zip(y_pol,x_pol) if y_lim[0] <= ia <= y_lim[1]])
                
                # Store the interpolated trimmed sequence obtained for this
                # cluster in final list.
                final_zams.append([x_pol_trim, y_pol_trim])
                # Also store the parameters associated with this cluster.
                final_zams_params.append([cluster, cl_e_bv, cl_age, cl_feh,
                                          cl_dmod])
            else:
                x_pol_trim, y_pol_trim = [], []

        
            # Write interpolated sequence to output file.
            write_seq_file(out_dir, cluster, x_pol_trim, y_pol_trim)
        
            # Call function to create CMDs for this cluster.
            m_c_c(sub_dir, cluster, col1_data, mag_data, stars_out_rjct,
                  stars_out, stars_in_rjct, stars_in, prob_memb_avrg,
                  popt_mag, popt_col1, cl_e_bv, cl_age, cl_feh, cl_dmod,
                  iso_moved, iso_intrsc, zams_iso, col1_min_int, col1_max_int, 
                  mag_min_int, mag_max_int, min_prob, fine_tune, x, y, kde,
                  manual_levels, col_intrsc, mag_intrsc, memb_above_lim,
                  zam_met, x_pol_trim, y_pol_trim, out_dir)
        


# Call to plot the 3 metallicty ranges.
print '\nPlotting sequences by metallicity interval'
# Define metallicity intervals to plot.
metal_ranges = [[-1.4, -0.71], [-0.7, -0.7], [-0.4, -0.4], [-0.3, -0.3],
                [-0.15, 0.01]]

# Create a plot for each metallicity range defined above.
for indx,m_rang in enumerate(metal_ranges):
    print 'Plotting %d' % indx
    m_f_p(indx, m_rang, zam_met, metals_z, metals_feh, final_zams_params,
          final_zams, out_dir)


print 'End.'