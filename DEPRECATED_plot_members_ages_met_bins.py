# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:10:04 2013

@author: gabriel
"""

from os import listdir, getcwd, walk
from os.path import join, realpath, dirname, getsize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator


'''
Take cluster member stars for a given set of clusters in a metallicty interval
and plot them in the intrinsec (corrected) CMD. Obtain from this plot the
approximated ZAMS making sure to leave out evolved stars.


* Files read by this code:

** All files with membership probabilities assigned to store the coordinates
   in CMD space and  weights of all the stars saved as most probable members:
   dir_memb_files + sub_dir + *_memb.dat

** File with the values obtained for each cluster after processing it with
   the OCAAT code.
   mypath + 'data_input'
      
** File with the parameters assigned to each cluster (by eye):
   mypath + 'clusters_data_isos.dat'

** File that stores a number of isochrones in the Washington system:
   mypath + zams_3_isos.data
   
   
* Files created by this code:

** Store corrected (intrinsec) values for magnitude and color for each cluster
   along with the rest of the data in a file according to a given metallicity
   interval.
   dir_memb_files  + memb_stars_metal_i.data
    
** Output PNG images of all clusters in a given metallicity interval
   superimposed.
   dir_memb_files + memb_stars_metal_i.png

'''



################### CHANGE THIS DIR ACCORDINGLY ###############################

# Location of the *_memb.dat output files.

dir_memb_files = '/home/gabriel/clusters/washington_KDE-Scott/'

###############################################################################


# Create output files for different intervals of metallicity.
for i in range(2):
    with open(dir_memb_files+'memb_stars_metal_%d.data' % (i+1), 'w') as f_out:
        f_out.write("#Name ID x y T1o e_T1 CT1o e_CT1 memb_index\n")


# Path where the code is running
mypath = realpath(join(getcwd(), dirname(__file__)))


# This list stores the sub dirs that should be excluded from the analysis.
exclude_refs = ['ref_33', 'ref_33_b']


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
    col_intrsc = col_obsrv - 1.97*e_bv
    mag_intrsc = mag_obsrv + 0.58*e_bv - dist_mod - 3.2*e_bv
    
    return col_intrsc, mag_intrsc




# Store subdir names [0] and file names [1] inside each subdir.
dir_files = [[], []]
for root, dirs, files in walk(dir_memb_files):
    if dirs:
        for subdir in dirs:
            for name in listdir(join(dir_memb_files, subdir)):
                # Check to see if it's a valid data file.
                if name.endswith(('_memb.dat')):
                    dir_files[0].append(subdir)
                    dir_files[1].append(name)
                    
                    
# Iterate through all files inside all sub-dirs.
for f_indx, sub_dir in enumerate(dir_files[0]):
    
    # Store cluster's name (with no .dat extension).
    clust_name = dir_files[1][f_indx][:-9]
    
    # Get contamination index for this cluster.
    with open(join(mypath, 'data_input'), mode="r") as f_cl_dt:
        for line in f_cl_dt:
            li=line.strip()
            # Jump comments.
            if not li.startswith("#"):
                reader = li.split()            
                # reader[4]= cont_index.
                if reader[0] == clust_name:
                    cont_indx = float(reader[4])
                    

    # Loads the data in file as a list of N lists where N is the number
    # of columns. Each of the N lists contains all the data for the column.
    file_path = join(dir_memb_files, sub_dir, clust_name+'_memb.dat')
    
    # Check if file is not empty. 44 bytes is the length of a file with only
    # the first line written on it.
    flag_1 = True if getsize(file_path) > 44 else False
    # Check if cluster is not in one of the rejected sub-folders.
    flag_2 = True if sub_dir not in exclude_refs else False
    # Check if the contamination index is low enough to accept the cluster.
    flag_3 = True if cont_indx < 0.6 else False
    # Check if the contamination index is low enough to accept the cluster.
    flag_4 = True if clust_name in piatti_accept else False
    
    if flag_1 and flag_2 and flag_3 and flag_4:
        
        print sub_dir+'/'+clust_name, 'accepted.'
        
        data0 = np.loadtxt(file_path, unpack=True)
        # Only store averaged member stars.
        data_filter = map(list, zip(*(col for col in zip(*data0) if col[0] == 99.)))
        
        # Get extinction and distance modulus values for this cluster.
        with open(join(mypath, 'clusters_data_isos.dat'), mode="r") as f_cl_dt:
    
            for line in f_cl_dt:
                li=line.split()
                # li[0]=name, [1]=E(B-V), [2]=Age, [3]=[Fe/H], [4]=(m-M)o
                if li[0] == clust_name:
                    # Obtain E(B-V), Age, [Fe/H] and (m-M)o values for this cluster.
                    cl_e_bv, cl_age, cl_feh, cl_dmod = float(li[1]), float(li[2]),\
                    float(li[3]), float(li[4])
        
        # Call function to move mag and color to intrinsic CMD.
        col_intrsc, mag_intrsc = intrsc_values(np.array(data_filter[6]),
                                               np.array(data_filter[4]),
                                               cl_e_bv, cl_dmod)
        
        # Replace observed mag and colors with intrinsic values.
        data_filter[4] = mag_intrsc.tolist()
        data_filter[6] = col_intrsc.tolist()
        
        # Replace average ID column (which contains only 99. values) with the
        # cluster's name.
        data_filter[0] = [clust_name for _ in range(len(data_filter[0]))]
    
    
        # Select metallicity interval.
#        if cl_feh < -1.:
#            met_interv = 1
#        elif -1 <= cl_feh < -0.75:
#            met_interv = 2
#        elif -0.75 <= cl_feh < -0.65:
#            met_interv = 3
#        elif -0.65 <= cl_feh <= -0.4:
#            met_interv = 4
#        elif -0.4 < cl_feh:
#            met_interv = 5


        # Store in the first file ALL the clusters, regardless of their
        # metallicity value.
        met_interv = 1
        # Output data file. Stores the names, parameters and member stars for each
        # cluster processed. Create one of this for each metallicity interval.
        out_data = dir_memb_files+'memb_stars_metal_%s.data' % str(met_interv)
        
        line = zip(*data_filter)

        with open(out_data, 'a') as f_out:
            for item in line:
                f_out.write('{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} \
{:>10} {:>10}'.format(*item))
                f_out.write('\n')

        # Store in second file only those with [Fe/H] > -0.75
        if cl_feh > -0.75:
            met_interv = 2
            # Output data file. Stores the names, parameters and member stars for each
            # cluster processed. Create one of this for each metallicity interval.
            out_data = dir_memb_files+'memb_stars_metal_%s.data' % str(met_interv)
            
            line = zip(*data_filter)
    
            with open(out_data, 'a') as f_out:
                for item in line:
                    f_out.write('{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} \
    {:>10} {:>10}'.format(*item))
                    f_out.write('\n')            

    else:
        print sub_dir+'/'+clust_name, 'rejected: ', flag_1, flag_2, flag_3, flag_4


# Get ZAMS.
file_path = join(mypath, 'zams_3_isos.data')
data = np.loadtxt(file_path, unpack=True)

# Convert z to [Fe/H] using the y=A+B*log10(x) zunzun.com function and the
# x,y values:
#   z    [Fe/H]
# 0.001  -1.3
# 0.004  -0.7
# 0.008  -0.4
# 0.019  0.0
#    A, B = 1.7354259305164, 1.013629121876
#    feh = A + B*np.log10(params[best_func][0])

# [Fe/H] = -1.15 --> z = 0.0014
# [Fe/H] = -0.875 --> z = 0.0027
# [Fe/H] = -0.7 --> z = 0.004
# [Fe/H] = -0.45 --> z = 0.007
# [Fe/H] = -0.2 --> z = 0.0123

#metals_z = [0.0014, 0.0027, 0.004, 0.007, 0.0123]
#metals_feh = [-1.15, -0.875, -0.7, -0.45, -0.2]
metals_z = [0.001, 0.004, 0.019]
metals_feh = [-1.3, -0.7, 0.0]

# List that holds all the isochroned of different metallicities.
zam_met = [[] for _ in range(len(metals_z))]

# Store each isochrone of a given metallicity in a list.
for indx, metal_val in enumerate(metals_z):
    zam_met[indx] = map(list, zip(*(col for col in zip(*data) if\
    col[0] == metal_val)))

        
# Make plots for each metallicity interval.
for i in range(2):
    interv = i+1
    
    # Read file where members are stored for all clusters.
    file_path = dir_memb_files+'memb_stars_metal_%s.data' % str(interv)
    data0 = np.genfromtxt(file_path, dtype=None, unpack=True)
    # Setting dtype=None apparently makes unpack=True not work anymore.
    data0 = zip(*data0)
    
    # Store names of clusters in this file excluding repeated entries by
    # using the 'set' function.
    cl_names_0 = data0[0]
    cl_names = list(set(cl_names_0))
    num_clusters = len(cl_names)


    # Store color, magnitude and weights into array.
    data = np.array([[data0[6], data0[4], data0[8]]], dtype=float)
    
    
    # figsize(x1, y1), GridSpec(y2, x2) --> To have square plots: x1/x2 = 
    # y1/y2 = 2.5 
    fig = plt.figure(figsize=(20, 30)) # create the top-level container
    gs = gridspec.GridSpec(12, 8)  # create a GridSpec object
    
    
    # Plot most probable members stars.
    ax0 = plt.subplot(gs[4:10, 2:6])
    plt.xlim(max(-0.9,min(data[0][0])-0.2), min(3.9, max(data[0][0])+0.2))
    plt.ylim(max(data[0][1])+1., min(data[0][1])-0.5)
    #Set axis labels
    plt.xlabel(r'$(C-T_1)_o$', fontsize=18)
    plt.ylabel(r'$M_{T_1}$', fontsize=18)

    # Set minor ticks and grid.
    ax0.minorticks_on()
    ax0.xaxis.set_major_locator(MultipleLocator(1.0))
    ax0.grid(b=True, which='major', color='gray', linestyle='--', lw=1)
     
    
    # Create new list with inverted values so higher prob stars are on top.
    m_p_m_temp = [data[0][0], data[0][1], data[0][2]]
    cm = plt.cm.get_cmap('RdYlBu_r')

    # Sort this list first by the decontamination index from max
    # value (1) to min (0) and then by its magnitude value, from min to max.
    m_p_m_temp_sort = zip(*sorted(zip(*m_p_m_temp), key=lambda x: x[2]))

    # Plot stars.
    plt.scatter(m_p_m_temp_sort[0], m_p_m_temp_sort[1], marker='o', 
                c=m_p_m_temp_sort[2], s=15, cmap=cm, lw=0.35, vmin=0, vmax=1)
    
    # Legend.            
#    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none',
#                      linewidth=0, label='Num clusters: %d' % num_clusters)


    # Add text box
    text = 'All clusters (%d)' % num_clusters if interv==1 else\
    'Clusters with [Fe/H]>-0.75 (%d)' % num_clusters
    xalign = 0.37 if interv==1 else 0.22
    plt.text(xalign, 0.97, text, transform=ax0.transAxes,
             bbox=dict(facecolor='white', alpha=0.6), fontsize=20)

    # Plot ZAMS envelope.
#    j = interv-1
    lin_st = ['-.', '--', '-']
    lin_w = [3., 2.5, 2.]
    for j in range(len(metals_z)):
        text1 = 'z = %0.4f (ZAMS)' '\n' % metals_z[j]
        text2 = '[Fe/H] = %0.3f' '\n' % metals_feh[j]
        text = text1+text2
        plt.plot(zam_met[j][3], zam_met[j][2], c='k', ls=lin_st[j], lw=lin_w[j],
                 label=text)
    
    # Plot legend.
    plt.legend(loc="upper right", markerscale=0.7, scatterpoints=1, fontsize=12)
    
    
    
    fig.tight_layout()
    
    out_png = join(dir_memb_files, 'memb_stars_metal_%d.png' % interv)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print 'Plot %d done.' % interv
 
    
print '\nEnd.'