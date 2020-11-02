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
import SimpleITK as sitk
from radiomics import featureextractor, getTestCase, firstorder
from collections import defaultdict

#function to get a dictionary of the variable names and indexes
def get_dict_var_names(data):
    data_var_names = data.dtype.base.base.fields.keys()
    dict_var_data = dict()
    for i, var in enumerate(data_var_names):
        dict_var_data[var] = i
    return dict_var_data

#define a funcion to normalize data within Patient/PEEP Trial/PEEP Level
def norm_PEEP(data, variable, level='Trial'):
    #Inter-quantile Normalization
    q5 = 0  #quantile 5%
    q95 = 0 #quantile 95%
    data_norm = np.copy(data)
    if level=='Trial':
        #finding smallest and biggest variable values
        for i in range(data.shape[0]):
            if i==0:
                q5 = np.quantile(data[i][variable],0.05)
                q95 = np.quantile(data[i][variable],0.95)
            else:
                if np.quantile(data[i][variable],0.05) < q5:
                    q5 = np.quantile(data[i][variable],0.05)
                if np.quantile(data[i][variable],0.95) > q95:
                    q95 = np.quantile(data[i][variable],0.95)
        #computing the value's range
        var_range = abs(q5)+abs(q95)
        #normalize through division of every value by range
        for i in range(data.shape[0]):
            data_norm[i][variable] = data[i][variable].copy()/var_range
        return data_norm

    elif level=='Level':
        #finding smallest and biggest variable values
        for i in range(data.shape[0]):
            q5 = np.quantile(data[i][variable],0.05)
            q95 = np.quantile(data[i][variable],0.95)
            #computing the value's range
            var_range = abs(q5)+abs(q95)
            #normalize through division of every value by range
            data_norm[i][variable] = data[i][variable].copy()/var_range
        return data_norm

    elif level=='Patient':
        pass #Here goes the code for intra-patient/extra/PEEP-trial normalization
    else:
        print('The input inserted for the argument level is not valid. Please choose one of the following options: \'Trial\', \'Level\', \'Patient\'.')
        raise ValueError

#define a funcion to get the img corresponding to the difference between 2 consecutive images
def img_diff(data, variable):
    data_diff = np.copy(data)
    for i in range(data.shape[0]):
        #Compute the differentiation between 2 consecutive images and extract features
        for j in range(data[i][variable].shape[0]):
            if j==0:
                img1=data[i][variable][j]
                continue
            img2=np.copy(img1)
            img1=data[i][variable][j]

            data_diff[i][variable][j] = img2-img1
    return data_diff

def feature_extraction(file,params,img_type='Original'):
    '''
    file: Patient file path
    img_type: Original/PEEPLevelNorm('Level')/PEEPTrialNorm('Trial')/PatientNorm('Patient')/Diff
    params: pyRadiomics params file path
    '''
    #load data
    data = io.loadmat(file)
    #extracting the structure SECTION
    section = data['section'][0]
    #Get a dictionary of the variable names
    dict_var_section = get_dict_var_names(section)
    #Instanciate the feature extration
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    #Create a (main) Pandas DateFrame to save the results
    RESULT = pd.DataFrame()
    # Transform (Normalize) dzMov
    if img_type!='Original':
        if img_type=='Diff':
            section = img_diff(section,dict_var_section['dzMov'])
        else:
            section = norm_PEEP(section,dict_var_section['dzMov'],img_type)
    #feature extraction
    for i in range(section.shape[0]):
        #Get PEEP level
        PEEPlvl=section[i][dict_var_section['PEEP']][0][0]
        #Get PEEP level lung shape
        lung_Shape = np.reshape(section[i][dict_var_section['lLungShape32']],(32,32)).copy()
        lung_Shape = sitk.GetImageFromArray(lung_Shape)
        #Create a dict to save the features
        d = defaultdict(list)
        #Compute the differentiation between 2 consecutive images and extract features
        for j in range(section[i][dict_var_section['dzMov']].shape[0]):
            img = np.reshape(section[i][dict_var_section['dzMov']][j],(32,32)).copy()
            img = sitk.GetImageFromArray(img)
            result = pd.Series(extractor.execute(img,lung_Shape))
            for k,v in result.items():
                    d[k].append(v)
        #convert d to a Pandas DataFrame and export it to a .csv file
        d = pd.DataFrame(d)
        d['PEEP'] = PEEPlvl
        d_agg = d.aggregate('mean', axis='rows')
        RESULT = pd.concat([RESULT,d_agg],axis=1)

        # d.to_csv(os.path.join(file_path, "features",f"Patient_XPTO_Trial{str(int(PEEPlvl))}.csv"))
        # d_agg.T.to_csv(os.path.join(file_path, "features",f"Agg_Patient_XPTO_Trial{str(int(round(PEEPlvl)))}.csv"))
    RESULT.T.to_csv(os.path.join(file_path, "features",f"Agg_Patient_XPTO_Trial{str(int(round(PEEPlvl)))}.csv"), index=False)

#main
file_path = r'C:\Users\Vasco\Documents\Ms Data Science\3Semester\Code\EIT_CoViD_Internship'
file_name = os.path.join(file_path,'Sample_EIT_Data',"COV077_05.mat")
params = os.path.join(file_path, "ParamsSettings_Pyradiomics_Params.yaml")
feature_extraction(file=file_name, params=params, img_type='Diff')