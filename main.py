import os
from helper_func import *
import multiprocessing as mp


#main
if __name__ == "__main__":
    pool = mp.Pool(processes=mp.cpu_count())
    file_path = r"C:\Users\Vasco\Documents\Ms Data Science\3Semester\Code\EIT_CoViD_Internship"
    params = os.path.join(file_path, "ParamsSettings_Pyradiomics_Params.yaml")
    #loop through all the files to get the .mat files and preform the radiomics feature extraction
    samples_folder = r"C:\Users\Vasco\Documents\Ms Data Science\3Semester\DATA\COVID2\MatLab Files\Supine\Batch2"
    processes = []
    for filename in os.listdir(samples_folder):
        if filename.endswith(".mat"):
            file_name = os.path.join(samples_folder,filename)
            pool.apply_async(feature_extraction, args=(file_name, params, None, 'Trial'))
        else:
            continue

    pool.close()
    pool.join()