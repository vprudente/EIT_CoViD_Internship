import os
from helper_func import *


#main
if __name__ == "__main__":
    file_path = r"C:\Users\Vasco\Documents\Ms Data Science\3Semester\Code\EIT_CoViD_Internship"
    params = os.path.join(file_path, "ParamsSettings_Pyradiomics_Params.yaml")
    #loop through all the files to get the .mat files and preform the radiomics feature extraction
    samples_folder = os.path.join(file_path,'Sample_EIT_Data')
    for filename in os.listdir(samples_folder):
        if filename.endswith(".mat"):
            file_name = os.path.join(samples_folder,filename)
            feature_extraction(file=file_name, params=params, img_type='Trial')
        else:
            continue