import os
import numpy as np
import random
COOKED_TRACE_FOLDER = './test/fcc/'

def load_trace(cooked_trace_folder, load_mahimahi_ptrs=True):
    cooked_trace_folder = '../Design'+cooked_trace_folder
    print("Loading traces from " + cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    all_mahimahi_ptrs = []

    para = 1 
    for subdir ,dirs ,files in os.walk(cooked_trace_folder):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        files.sort()
        for file in files:
            file_path = subdir + os.sep + file
            if file_path.endswith('.pkl'):
                continue

            val_folder_name = os.path.basename( os.path.normpath( subdir ) )
            cooked_time = []
            cooked_bw = []
            with open(file_path, 'rb') as phile:
                for line in phile:
                    parse = line.split()
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]))
            length = len(cooked_bw) 
            all_cooked_time.append(cooked_time[:length])
            all_cooked_bw.append(cooked_bw[:length])
            all_file_names.append( file)
    print(len(all_cooked_bw[-1]))

    return all_cooked_time, all_cooked_bw, all_file_names