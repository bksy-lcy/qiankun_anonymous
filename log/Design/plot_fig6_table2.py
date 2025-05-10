import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig
import matplotlib
import sys
from collections import OrderedDict
import scipy.stats
plt.switch_backend('agg')

NUM_BINS = 500
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
K_IN_M = 1000.0
REBUF_P = 4.3
SMOOTH_P = 1

LW = 4

d_values =  [f"{round(i*0.1+0.1, 1):.1f}" for i in range(10)]
SCHEMES = ['_bb_', '_rl_', 'mpc', 'hyb','bola','sim_ghent_','netllm','5llm']
labels = ['BBA', 'Pensieve', 'RobustMPC', 'HYB', 'BOLA', 'Genet','NetLLM','QianKun']
markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
lines = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
modern_academic_colors = ['#4E79A7', '#F28E2B', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7','#E15759','#DAA520']
alg_sch_all = {}
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def inlist(filename, traces):
    ret = False
    for trace in traces:
        if trace in filename:
            ret = True
            break
    return ret

def bitrate_smo(outputs,LOG):
   
    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[1:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[1:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_bit)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_smo)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            color = modern_academic_colors[idx],
            marker = markers[idx], markersize = 10, label = labels[idx],
            capsize=4)

        out_str = '%s %.3f %.3f %.3f %.3f %.3f %.3f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Bitrate Smoothness (mbps)')
    ax.set_ylabel('Video Bitrate (mbps)')
    ax.set_ylim(max_bitrate * 0.5, max_bitrate * 1.01)

    ax.grid(linestyle='--', linewidth=1., alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower left')
    ax.invert_xaxis()

    fig.savefig(outputs + '.png')
    plt.close()

def smo_rebuf(outputs,LOG):

    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[1:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[1:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_smo)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_rebuf)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            color = modern_academic_colors[idx],
            marker = markers[idx], markersize = 10, label = labels[idx],
            capsize=4)

        out_str = '%s %.3f %.3f %.3f %.3f %.3f %.3f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Time Spent on Stall (%)')
    ax.set_ylabel('Bitrate Smoothness (mbps)')
    ax.set_ylim(0.05, max_bitrate + 0.05)

    ax.grid(linestyle='--', linewidth=2.,)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower left')
    ax.invert_xaxis()
    ax.invert_yaxis()

    fig.savefig(outputs + '.png')
    plt.close()

def bitrate_rebuf(outputs,LOG):
    
    reward_all = {}

    plt.rcParams['axes.labelsize'] = 18
    font = {'size': 18}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plt.subplots_adjust(left=0.12, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scheme = LOG + '/' + files
                f = open(file_scheme, 'r')
                lines = f.readlines()
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in lines:
                    sp = line.split('\t')
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[1:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[1:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_bit)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_rebuf)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            color = modern_academic_colors[idx],lw=LW-0.5,
            marker = markers[idx], markersize = 15, label = labels[idx],
            capsize=4)

        out_str = '%s %.3f %.3f %.3f %.3f %.3f %.3f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Time Spent on Stall (%)')
    ax.set_ylabel('Video Bitrate (mbps)')
    ax.set_ylim(max_bitrate * 0.5, max_bitrate * 1.01)

    ax.grid(linestyle='--', linewidth=2.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=18, ncol=3, edgecolor='white',loc='lower left')
    ax.invert_xaxis()

    fig.savefig(outputs + '.pdf')
    plt.close()

def qoe_cdf(outputs,LOG):
    reward_all = {}
    
    plt.rcParams['axes.labelsize'] = 18
    font = {'size': 18}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plt.subplots_adjust(left=0.07, bottom=0.16, right=0.96, top=0.96)

    # Used to record the minimum and maximum values of all data
    min_value = float('inf')
    max_value = float('-inf')
    
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        arr.append(float(sp[-1]))
                f.close()
                mean_arr.append(np.mean(arr[1:]))
        
        reward_all[scheme] = mean_arr
        
        # Update the minimum and maximum values
        min_value = min(min_value, min(mean_arr))
        max_value = max(max_value, max(mean_arr))

        values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
        cumulative = np.cumsum(values)
        cumulative = cumulative / np.max(cumulative)
        ax.plot(base[:-1], cumulative, '-', \
                color=modern_academic_colors[idx], lw=LW, \
                label='%s' % (labels[idx]))

        print('%s, %.3f' % (scheme, np.mean(mean_arr)),np.std(mean_arr))
        if scheme not in alg_sch_all:
            alg_sch_all[scheme] = mean_arr
        else:
            alg_sch_all[scheme] += mean_arr
    # Set the x-axis range: if the minimum value is greater than 0, start from the minimum value, otherwise start from 0
    x_min = min_value if min_value > 0 else 0
    x_max = max_value
    
    ax.set_xlabel('QoE')
    ax.set_ylim(0., 1.01)
    ax.set_xlim(x_min, x_max)

    ax.grid(linestyle='--', linewidth=2.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=18, ncol=3, edgecolor='white',loc='lower right')

    fig.savefig(outputs + '.pdf')
    plt.close()
def find_best_qoe_per_trace(LOG):
    # Dictionary to store QoE values for each trace and algorithm
    trace_algo_qoe = {}
    
    # Process all files in the log directory
    for filename in os.listdir(LOG):
        for scheme in SCHEMES:
            if scheme in filename and 'llm' not in filename:
                # Extract trace name from filename
                # Assuming filename format contains trace name
                trace_name = filename.replace(scheme, '').strip('_')
                
                # Read QoE values
                with open(os.path.join(LOG, filename), 'r') as f:
                    arr = []
                    for line in f:
                        sp = line.split()
                        if len(sp) > 1:
                            arr.append(float(sp[-1]))
                    
                    # Calculate mean QoE (excluding first value)
                    mean_qoe = np.mean(arr[1:])
                    
                    # Store in dictionary
                    if trace_name not in trace_algo_qoe:
                        trace_algo_qoe[trace_name] = {}
                    trace_algo_qoe[trace_name][scheme] = mean_qoe

    # Find and print best algorithm for each trace
    print("\nBest performing algorithms per trace:")
    print("-" * 50)
    for trace_name in sorted(trace_algo_qoe.keys()):
        # Find algorithm with highest QoE for this trace
        best_algo = max(trace_algo_qoe[trace_name].items(), key=lambda x: x[1])
        algo_name = labels[SCHEMES.index(best_algo[0])]  # Convert scheme name to label
        
        print(f'Trace: {trace_name}')
        print(f'Algorithm: {algo_name} (QoE: {best_algo[1]:.3f})')
        print("-" * 50)

    # Optional: Print overall statistics
    print("\nOverall best performing algorithms count:")
    algo_counts = {}
    for trace_data in trace_algo_qoe.values():
        best_algo = max(trace_data.items(), key=lambda x: x[1])[0]
        algo_counts[best_algo] = algo_counts.get(best_algo, 0) + 1
    
    for scheme, count in algo_counts.items():
        algo_name = labels[SCHEMES.index(scheme)]
        print(f'{algo_name}: {count} traces')
def analyze_by_file(LOG):
    file_results = {}  # Store results by filename
    best_schemes = {}  # Best scheme for each file
    test_lists = os.listdir(LOG)
    for file in test_lists:
        file_results[file] = {}
    # Traverse all files
    for file in os.listdir(LOG):
        # Find the corresponding scheme for the file
        # Search for matching basename in test_lists
        base_name = None
        for test_file in test_lists:
            
            if test_file in file:  # If the test filename is a substring of LOG filename
                base_name = test_file
                break
        print(file,base_name)
        for scheme in SCHEMES:
            if scheme in file:
                file_path = os.path.join(LOG, file)
                with open(file_path, 'r') as f:
                    arr = []
                    for line in f:
                        sp = line.split()
                        if len(sp) > 1:
                            arr.append(float(sp[-1]))
                    
                    if arr:  # Ensure there is data
                        mean_value = np.mean(arr[1:])
                        file_results[base_name][scheme] = mean_value
    
    # Find the optimal scheme for each file type
    for file_type, schemes in file_results.items():
        if schemes:  # Ensure this file type has data
            best_scheme = max(schemes.items(), key=lambda x: x[1])
            best_schemes[file_type] = best_scheme
            
            print(f"\nFile type: {file_type}")
            print("Results for each scheme:")
            for scheme, value in schemes.items():
                print(f"{scheme}: {value:.2f}")
            print(f"Best scheme: {best_scheme[0]} with value: {best_scheme[1]:.2f}")
    
    return file_results, best_schemes
def plot_bar(outputs, LOG):
    # Initialize data storage
    data_means = {scheme: {'qoe': [], 'bitrate': [], 'smooth': [], 'rebuf': []} for scheme in SCHEMES}
    
    # Set chart style
    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(12, 6))  # Widen the figure to accommodate 4 metrics
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.85)
    
    # Define fill styles
    hatchs = ['++++++', None,  '\\\\\\\\\\','xxxxxxxx', '\\\\\\\\\\', '***', 'ooooo', 'OOOOO', 
              '....', '+++++', '////', '****', 'ooooo', 'OOOOO', '....', '+++++', '////', '****', 
              'ooooo', 'OOOOO', '....', '+++++', '////', '****']
    
    # Data collection
    for scheme in SCHEMES:
        for files in os.listdir(LOG):
            if scheme in files:
                file_path = os.path.join(LOG, files)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < 2:
                        continue
                        
                    bitrates = []
                    rebuf_times = []
                    qoe_values = []
                    
                    for line in lines:
                        sp = line.split('\t')
                        if len(sp) > 1:
                            bitrates.append(float(sp[1]) / 1000.0)  # Convert to Mbps
                            rebuf_times.append(float(sp[3])*4.3)
                            qoe_values.append(float(sp[-1]))
                    
                    # Calculate average values, smoothness penalty, and rebuffering penalty
                    data_means[scheme]['bitrate'].append(np.mean(bitrates))
                    data_means[scheme]['qoe'].append(np.mean(qoe_values))
                    data_means[scheme]['smooth'].append(np.mean(np.abs(np.diff(bitrates))))
                    data_means[scheme]['rebuf'].append(np.mean(rebuf_times[1:]))
    
    # Plot settings
    bar_width = 0.15
    opacity = 0.8
    index = np.arange(4)  # 4 metrics: QoE, Bitrate, Smooth, Rebuf
    
    # Draw bar chart
    for idx, scheme in enumerate(SCHEMES):
        means = [
            np.mean(data_means[scheme]['qoe']),
            np.mean(data_means[scheme]['bitrate']),
             np.mean(data_means[scheme]['rebuf']),
            np.mean(data_means[scheme]['smooth']),
           
        ]
        
        # Calculate error intervals
        errs = [
            mean_confidence_interval(data_means[scheme]['qoe'])[2] - mean_confidence_interval(data_means[scheme]['qoe'])[0],
            mean_confidence_interval(data_means[scheme]['bitrate'])[2] - mean_confidence_interval(data_means[scheme]['bitrate'])[0],
             mean_confidence_interval(data_means[scheme]['rebuf'])[2] - mean_confidence_interval(data_means[scheme]['rebuf'])[0],
            mean_confidence_interval(data_means[scheme]['smooth'])[2] - mean_confidence_interval(data_means[scheme]['smooth'])[0],
           
        ]
        
        # Draw bar chart and error bars
        ax.bar(index + idx * bar_width, means, bar_width,
               alpha=opacity,
               color='white',  # Set white background
               edgecolor=modern_academic_colors[idx],  # Border color
               hatch=hatchs[idx],  # Add fill style
               label=labels[idx],
               yerr=errs,
               error_kw={'elinewidth': 2, 'capsize': 5},
               linewidth=1.5)  # Increase border width to make fill style more clear
    
    # Set chart properties
    ax.set_ylabel('Values')
    ax.set_xticks(index + bar_width * (len(SCHEMES) - 1) / 2)
    ax.set_xticklabels(['QoE', 'Bitrate (Mbps)', 'Rebuffer Penalty', 'Smooth Penalty'])
    
    # Add grid and legend
    ax.grid(linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=15, ncol=5, loc='upper center')
    
    # Save chart
    plt.savefig(outputs + '.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    lists = os.listdir('results')
    for file in ["3g","5g", "ghent", "hsr" ,"lab" , "fcc-test" ,"fcc-train","oboe","puffer-2202","fcc18"]:
        print(file)
        LOG = 'results/'+file
        qoe_cdf('img/'+file,LOG)
print('avg')
for scheme in SCHEMES:
    print(f"{scheme} : {np.mean(alg_sch_all[scheme])}")