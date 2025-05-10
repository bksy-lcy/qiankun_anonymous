import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean
from matplotlib.ticker import MaxNLocator
import matplotlib

def ewma(data, alpha):
    """Calculate Exponentially Weighted Moving Average"""
    result = [data[0]]  # Initialize with first value
    for n in range(1, len(data)):
        result.append(alpha * data[n] + (1 - alpha) * result[n-1])
    return result

def calculate_errors(log_file_path):
    # Read data from log file
    bandwidth_data = []
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                if len(items) < 8:  # Ensure there are enough columns
                    continue
                try:
                    past_bw = float(items[6])
                    future_bw = float(items[7])
                    bandwidth_data.append((past_bw, future_bw))
                except (ValueError, IndexError):
                    continue
        
        if len(bandwidth_data) < 6:  # Ensure there are enough data points
            return None
        
        # Calculate errors
        prediction_errors = []
        avg_prediction_errors = []
        harmonic_prediction_errors = []
        ewma_prediction_errors = []
        
        # EWMA parameters
        alpha = 0.125  # Smoothing factor
        
        for i in range(5, len(bandwidth_data)):
            # Current past_bw vs previous future_bw
            actual = bandwidth_data[i][0]  # Current past_bw
            predicted = bandwidth_data[i-1][1]  # Previous future_bw
            error = actual - predicted
            prediction_errors.append(error)
            
            # Average of past 5 bandwidth vs previous future_bw
            start_idx = max(0, i-5)
            past_bws = [x[0] for x in bandwidth_data[start_idx:i]]
            avg_past_bw = np.mean(past_bws)
            avg_error = actual - avg_past_bw
            avg_prediction_errors.append(avg_error)
            
            # Harmonic mean of past 5 bandwidth vs previous future_bw
            positive_past_bws = [abs(x) for x in past_bws]
            if all(bw > 0 for bw in positive_past_bws):
                harmonic_past_bw = hmean(positive_past_bws)
                if sum(1 for x in past_bws if x < 0) > len(past_bws)/2:
                    harmonic_past_bw = -harmonic_past_bw
            else:
                harmonic_past_bw = avg_past_bw
                
            harmonic_error = actual - harmonic_past_bw
            harmonic_prediction_errors.append(harmonic_error)
            
            # EWMA prediction
            ewma_past_bws = ewma(past_bws, alpha)
            ewma_pred = ewma_past_bws[-1]
            ewma_error = actual - ewma_pred
            ewma_prediction_errors.append(ewma_error)
        
        # Calculate error metrics
        metrics = {
            'Single Prediction': {
                'MAE': np.mean(np.abs(prediction_errors)),
                'MSE': np.mean(np.square(prediction_errors)),
                'MAPE': np.mean(np.abs(np.array(prediction_errors) / np.array([x[0] for x in bandwidth_data[5:]]))) * 100,
                'raw_errors': prediction_errors
            },
            'Arithmetic Mean Prediction': {
                'MAE': np.mean(np.abs(avg_prediction_errors)),
                'MSE': np.mean(np.square(avg_prediction_errors)),
                'MAPE': np.mean(np.abs(np.array(avg_prediction_errors) / np.array([x[0] for x in bandwidth_data[5:]]))) * 100,
                'raw_errors': avg_prediction_errors
            },
            'Harmonic Mean Prediction': {
                'MAE': np.mean(np.abs(harmonic_prediction_errors)),
                'MSE': np.mean(np.square(harmonic_prediction_errors)),
                'MAPE': np.mean(np.abs(np.array(harmonic_prediction_errors) / np.array([x[0] for x in bandwidth_data[5:]]))) * 100,
                'raw_errors': harmonic_prediction_errors
            },
            'EWMA Prediction': {
                'MAE': np.mean(np.abs(ewma_prediction_errors)),
                'MSE': np.mean(np.square(ewma_prediction_errors)),
                'MAPE': np.mean(np.abs(np.array(ewma_prediction_errors) / np.array([x[0] for x in bandwidth_data[5:]]))) * 100,
                'raw_errors': ewma_prediction_errors
            }
        }
        return metrics
    except Exception as e:
        print(f"Error processing file {log_file_path}: {str(e)}")
        return None

def plot_cdf_for_dataset(dataset_name, dataset_metrics, error_type, output_dir="./"):
    """Plot CDF of specified error type for a single dataset"""
    # Set chart style - white background
    plt.rcParams['axes.labelsize'] = 18
    font = {'size': 18}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)
    
    # Set color scheme
    colors = ['#F28E2B', '#4E79A7', '#59A14F', '#E15759']
    
    # Create chart
    labels = ['Mean', 'Harmonic', 'EWMA', 'QianKun']
    for pred_idx, pred_type in enumerate(['Arithmetic Mean Prediction', 
                                        'Harmonic Mean Prediction', 'EWMA Prediction', 'Single Prediction']):
        # Get error values from all files
        all_errors = []
        
        for file_metrics in dataset_metrics:
            if pred_type in file_metrics:
                if error_type == 'MAE':
                    # Extract absolute errors for MAE CDF
                    raw_errors = file_metrics[pred_type]['raw_errors']
                    errors = np.abs(raw_errors)
                else:  # MSE
                    # Extract squared errors for MSE CDF
                    raw_errors = file_metrics[pred_type]['raw_errors']
                    errors = np.square(raw_errors)
                    
                all_errors.extend(errors)
        
        # Plot CDF
        if all_errors:
            x = np.sort(all_errors)
            y = np.arange(1, len(x) + 1) / len(x)
            
            color = colors[pred_idx % len(colors)]
            
            # Calculate average metric value for this prediction type
            avg_value = np.mean(all_errors)
            label = f"{labels[pred_idx]}: {avg_value:.2f}"
            if dataset_name == 'puffer-2202':
                label = f"{labels[pred_idx]}: {avg_value:.4f}"
            
            ax.plot(x, y, label=label, color=color, linewidth=4)
    
    # Set chart properties
    ax.set_xlabel(error_type, fontsize=18)
    if dataset_name == 'puffer-2202':
        ax.set_xlim(0, 0.25)
        if error_type == 'MSE':
            ax.set_xlim(0, 0.06)
    if dataset_name == 'hsr':
        ax.set_xlim(0, 2)
        if error_type == 'MSE':
            ax.set_xlim(0, 3.5)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.01)
    ax.grid(True, linestyle='--', lw=2)
    
    # Add legend
    ax.legend(loc='lower right', fancybox=True)
    
    # Add arrow and English label in top left corner
    # Get x and y axis ranges
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    
    # Calculate arrow start and end positions (top left area)
    arrow_start_x = x_range[1] * 0.1  # 25% of x-axis
    arrow_start_y = y_range[1] * 0.85  # 80% of y-axis
    arrow_end_x = x_range[1] * 0.05   # Point toward upper left
    arrow_end_y = y_range[1] * 0.95   # Point toward upper left
    
    # Add larger arrow and English label
    ax.annotate('Better', 
                xy=(arrow_end_x, arrow_end_y),  # Arrow points to this position
                xytext=(arrow_start_x, arrow_start_y),  # Text position
                arrowprops=dict(facecolor='black', shrink=0.05, width=4, headwidth=12, headlength=10),
                fontsize=18, fontweight='bold')
    
    # Save chart
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{dataset_name}_{error_type}_cdf.pdf")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved {error_type} CDF plot for {dataset_name} to {output_file}")

def main():
    # Store metrics for different datasets
    datasets_metrics = {}
    
    # Create output directory (if it doesn't exist)
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Scan results directory
    results_dir = 'results'
    datasets = ['hsr', 'puffer-2202']
    
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        log_dir = os.path.join(results_dir, dataset)
        all_metrics = []
        
        for filename in os.listdir(log_dir):
            if '5llm' in filename:
                log_path = os.path.join(log_dir, filename)
                metrics = calculate_errors(log_path)
                
                if metrics is not None:
                    all_metrics.append(metrics)
        
        if all_metrics:
            datasets_metrics[dataset] = all_metrics
            
            # Calculate average metrics across all files
            print("=== Average Metrics Across All Files ===")
            avg_metrics = {}

            for pred_type in all_metrics[0].keys():
                avg_metrics[pred_type] = {}
                for metric_name in all_metrics[0][pred_type].keys():
                    if metric_name != 'raw_errors':  # Exclude raw error data
                        values = [m[pred_type][metric_name] for m in all_metrics]
                        avg_metrics[pred_type][metric_name] = np.mean(values)

            # Print average metrics
            for pred_type, error_metrics in avg_metrics.items():
                print(f"\n{pred_type}:")
                for metric_name, value in error_metrics.items():
                    print(f"{metric_name}: {value:.4f}")
            print('\n')
        else:
            print(f"No valid metrics found for dataset: {dataset}")
    
    # Plot separate CDF charts for each dataset and metric
    for dataset_name, dataset_metrics in datasets_metrics.items():
        for error_type in ['MAE', 'MSE']:
            plot_cdf_for_dataset(dataset_name, dataset_metrics, error_type, output_dir)

if __name__ == "__main__":
    main()