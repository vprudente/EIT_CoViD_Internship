import scipy.io as io
import os
import numpy as np
import cv2
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor, firstorder
from collections import defaultdict

#function to get a dictionary of the variable names and indexes
def get_dict_var_names(data):
    data_var_names = data.dtype.base.base.fields.keys()
    dict_var_data = dict()
    for i, var in enumerate(data_var_names):
        dict_var_data[var] = i
    return dict_var_data

#define a funcion to normalize data within Patient/PEEP Trial/PEEP Level
def norm_PEEP(data, variable, level='Trial', thresh=True):
    #Inter-quantile Normalization
    q1 = 0  #quantile 1%
    q99 = 0 #quantile 99%
    data_norm = np.copy(data)
    if level=='Trial':
        #finding smallest and biggest variable values
        for i in range(data.shape[0]):
            if i==0:
                q1 = np.quantile(data[i][variable],0.01)
                q99 = np.quantile(data[i][variable],0.99)
            else:
                if np.quantile(data[i][variable],0.01) < q1:
                    q1 = np.quantile(data[i][variable],0.01)
                if np.quantile(data[i][variable],0.99) > q99:
                    q99 = np.quantile(data[i][variable],0.99)
        #computing the value's range
        var_range = abs(q1)+abs(q99)
        #normalize through division of every value by range
        for i in range(data.shape[0]):
            # Threshold to contain only value between quantiles 1% and 99%
            if thresh:
                data_norm[i][variable] = np.clip(data_norm[i][variable], q1, q99)
            # Normalize
            data_norm[i][variable] = (data[i][variable].copy()-q1)/var_range
        return data_norm

    elif level=='Level':
        #finding smallest and biggest variable values
        for i in range(data.shape[0]):
            q1 = np.quantile(data[i][variable],0.01)
            q99 = np.quantile(data[i][variable],0.99)
            #computing the value's range
            var_range = abs(q1)+abs(q99)
            # Threshold to contain only value between quantiles 1% and 99%
            if thresh:
                data_norm[i][variable] = np.clip(data_norm[i][variable], q1, q99)
            #normalize through division of every value by range
            data_norm[i][variable] = (data[i][variable].copy()-q1)/var_range
        return data_norm

    elif level=='Patient':
        pass #Here goes the code for intra-patient/extra-PEEP-trial normalization
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

#define a function to extract the patient and PEEP trial code/name
def find_name( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def feature_extraction(file, params, out_path=None, img_type='Original'):
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
    #Find patient name and trial number
    name = find_name(file,os.path.dirname(file),'.mat')
    name = name.replace('\\','')
    name = name.split('_')
    if len(name)==1:
        name.append('00')
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
        # Dilate and erode the lung shape to remove eventual holes in the mask
        lung_Shape = cv2.dilate(lung_Shape, np.ones((2,2)))
        lung_Shape = cv2.erode(lung_Shape, np.ones((2,2)))
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
        d['PEEPLvl'] = PEEPlvl
        # d_agg = d.aggregate('mean', axis='rows')
        d['Frame'] = np.linspace(1, d.shape[0], d.shape[0])
        d['Patient'] = name[0]
        d['PEEPTrial_Num'] = name[1]
        RESULT = pd.concat([RESULT,d.T],axis=1)

        # Create a directory for the results file
        if out_path == None:
            file_parent = os.path.dirname(file)
            out_path = os.path.join(file_parent, "features")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        # d.to_csv(os.path.join(out_path,f"Patient_{name[0]}_Trial{name[1]}_PEEP{int(PEEPlvl)}.csv"))
        # d_agg.T.to_csv(os.path.join(out_path,f"Agg_Patient_{name[0]}_Trial{name[1]}_PEEP{int(PEEPlvl)}.csv"))
    # Export results
    RESULT.T.to_csv(os.path.join(out_path,f"Non-Agg_Patient_{name[0]}_Trial{name[1]}.csv"), index=False)
