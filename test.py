import scipy.io as io
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cv2
import pandas as pd
#from cmap_map import cmap_map

#importing data
file_path = 'Sample_EIT_Data/'
file_name = os.path.join(file_path,"COV077_05.mat")
data = io.loadmat(file_name)

#function to get a dictionary of the variable names and indexes
def get_dict_var_names(data):
    data_var_names = data.dtype.base.base.fields.keys()
    dict_var_data = dict()
    for i, var in enumerate(data_var_names):
        dict_var_data[var] = i
    return dict_var_data

#Getting variable names and indexes
dict_var_data = get_dict_var_names(data['data'][0])
dict_var_section = get_dict_var_names(data['section'][0])

#extracting the structure SECTION
section = data['section'][0]

#define a funcion to normalize data within Patient/PEEP Trial/PEEP Level
def norm_PEEP(data, variable, level='trial'):
    min_ = 0
    max_ = 0
    data_norm = np.copy(data)
    if level=='trial':
        #finding smallest and biggest variable values
        for i in range(data.shape[0]):
            if i==0:
                min_ = np.min(data[i][variable])
                max_ = np.max(data[i][variable])
            else:
                if np.min(data[i][variable])<min_:
                    min_ = np.min(data[i][variable])
                if np.max(data[i][variable])>max_:
                    max_ = np.max(data[i][variable])
        #computing the value's range
        var_range = abs(min_)+abs(max_)
        #normalize through division of every value by range?
        for i in range(data.shape[0]):
            data_norm[i][variable] = data[i][variable]/var_range
        return data_norm

    elif level=='level':
        #finding smallest and biggest variable values
        for i in range(data.shape[0]):
            min_ = np.min(data[i][variable])
            max_ = np.max(data[i][variable])
            #computing the value's range
            var_range = abs(min_)+abs(max_)
            #normalize through division of every value by range?
            data_norm[i][variable] = data[i][variable]/var_range
        return data_norm

    elif level=='patient':
        pass
    else:
        print('The input inserted for the argument level is not valid. Please choose one of the following options: \'trial\', \'level\', \'patient\'.')
        raise ValueError

# Normalize dzMov by PEEP Trial and Level
dzMov_norm = norm_PEEP(section,dict_var_section['dzMov'])
dzMov_norm2 = norm_PEEP(section,dict_var_section['dzMov'],'level')
#new_cmap = cmap_map(lambda x: 1-x, cm.seismic)

# Getting the idx corresponting to the dzMov variable
idx = dict_var_section['dzMov']
# Plotting the images for the original/PEEP-Trial/PEEP-Level normalized data
for i in range(dzMov_norm.shape[0]):
        fig1 = plt.figure('Original')
        fig2 = plt.figure('Normalized_PEEP_Trial')
        fig3 = plt.figure('Normalized_PEEP_Level')
        PEEPlvl=dzMov_norm[i][dict_var_section['PEEP']][0][0]
        fig1.suptitle('Original -> PEEP: '+str(PEEPlvl))
        fig2.suptitle('Normalized_PEEP_Trial -> PEEP: '+str(PEEPlvl))
        fig3.suptitle('Normalized_PEEP_Level -> PEEP: '+str(PEEPlvl))

        for j in range(dzMov_norm[i][idx].shape[0]):
            img = np.reshape(section[i][idx][j],(32,32))
            ax1 = fig1.add_subplot(5,section[i][idx].shape[0]/5+1,j+1)
            ax1.imshow(img, cmap='twilight', vmin=-1, vmax=1)
            ax1.axis('off')

            img = np.reshape(dzMov_norm[i][idx][j],(32,32))
            ax2 = fig2.add_subplot(5,dzMov_norm[i][idx].shape[0]/5+1,j+1)
            ax2.imshow(img, cmap='twilight', vmin=-1, vmax=1)
            ax2.axis('off')
            
            img = np.reshape(dzMov_norm[i][idx][j],(32,32))
            ax3 = fig3.add_subplot(5,dzMov_norm2[i][idx].shape[0]/5+1,j+1)
            ax3.imshow(img, cmap='twilight', vmin=-1, vmax=1)
            ax3.axis('off')
        plt.show()

# Plotting the images for the differentiation of 2 consecutive frames of PEEP-Trial normalized data
for i in range(dzMov_norm.shape[0]):
        fig4 = plt.figure('Diff_Norm')
        PEEPlvl=dzMov_norm[i][dict_var_section['PEEP']][0][0]

        for j in range(dzMov_norm[i][idx].shape[0]):
            if j==0:
                img1=section[i][idx][j]
                continue
            img2=np.copy(img1)
            img1=section[i][idx][j]

            img_diff = np.reshape(img2-img1,(32,32))
            ax4 = fig4.add_subplot(5,section[i][idx].shape[0]/5+1,j+1)
            ax4.imshow(img_diff, cmap='twilight', vmin=-0.01, vmax=0.01)
            ax4.axis('off')
        fig4.suptitle('Diif Norm -> PEEP: '+str(PEEPlvl))
        plt.show()

