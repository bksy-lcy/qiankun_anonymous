import subprocess
import sys
from bayes_opt import BayesianOptimization
import numpy as np

def generate_doubled_data(all_cooked_bw, all_cooked_time):
    # Usage Example
    # Assuming existing data
    # all_cooked_bw = [...]
    # all_cooked_time = [...]
    # Generate new data
    # new_bw, new_time = generate_doubled_data(all_cooked_bw, all_cooked_time)
    
    # Calculate the mean and variance of the bandwidth data
    bw_mean = np.mean(all_cooked_bw)
    bw_std = np.std(all_cooked_bw)
    
    # Calculate the time interval
    time_intervals = []
    for i in range(1, len(all_cooked_time)):
        time_intervals.append(all_cooked_time[i] - all_cooked_time[i-1])
    
    # Calculate the mean and variance of the time intervals
    interval_mean = np.mean(time_intervals)
    interval_std = np.std(time_intervals)
    
    # Generate new bandwidth data (double the quantity)
    new_bw = np.random.normal(bw_mean, bw_std, 2 * len(all_cooked_bw))
    
    # Ensure that the bandwidth data is positive
    new_bw = np.maximum(new_bw, 0)
    
    # Generate new time intervals (double the quantity minus one)
    new_intervals = np.random.normal(interval_mean, interval_std, 2 * len(all_cooked_bw) - 1)
    
    # Ensure that the time intervals are positive
    new_intervals = np.maximum(new_intervals, 0.01)  # Set the minimum interval to 0.01
    
    # Generate new time data starting from 0
    new_time = [0]
    for interval in new_intervals:
        new_time.append(new_time[-1] + interval)
    
    return new_bw.tolist(), new_time

def bayes(all_cooked_bw,all_cooked_time,buffer_init,video_init,discount_factor):
    all_cooked_bw,all_cooked_time = generate_doubled_data(all_cooked_bw, all_cooked_time)
    print("start bayes: ",np.mean(all_cooked_bw),np.std(all_cooked_bw),len(all_cooked_bw),buffer_init,video_init,discount_factor)
    def parse_result(result_str):
        """
        Parse the output results of the ABR algorithm.

        Args:
            result_str: A string containing multiple lists and parameters.
            
        Returns:
            Tuple[List[float], List[float], List[float], List[float], float, int]:
            - A list of Quality of Experience (QoE) values.
            - A list of bitrate selections.
            - A list of rebuffering events.
            - A list of smoothness penalties.
            - Buffer size.
            - Remaining number of video segments.
        """
        try:
            # Find the location of all square brackets
            brackets = []
            stack = []
            for i, char in enumerate(result_str):
                if char == '[':
                    stack.append(i)
                elif char == ']':
                    if stack:
                        start = stack.pop()
                        brackets.append((start, i))

            if len(brackets) != 4:
                raise ValueError(f"Expected 4 lists in result, found {len(brackets)}")

            # Parsing four lists
            def parse_list(start, end):
                list_str = result_str[start+1:end]
                return [float(x.strip()) for x in list_str.split(',') if x.strip()]

            r_batch = parse_list(*brackets[0])
            video_batch = parse_list(*brackets[1])
            rebuf_batch = parse_list(*brackets[2])
            smooth_batch = parse_list(*brackets[3])

            # Parsing the last two numbers
            remaining_str = result_str[brackets[-1][1]+1:].strip()
            buffer_size, chunks_remain = map(float, remaining_str.split())

            return r_batch, video_batch, rebuf_batch, smooth_batch, buffer_size, int(chunks_remain)

        except Exception as e:
            print(f"Result parsing failed: {e}")
            return [], [], [], [], 0, 0

    def objective_function(discount_factor):
        """
        Optimize the objective function
        
        Args:
            discount_factor: Discount factor parameter
            
        Returns:
            float: Average QoE value
        """
        try:
            bandwidth_series = str(all_cooked_bw).replace(' ', '')
            time_series = str(all_cooked_time).replace(' ', '')
            
            cmd = f'python m_pc.py {bandwidth_series} {time_series} {buffer_init} {video_init} {discount_factor}'
            
            result = subprocess.check_output(cmd, shell=True, text=True)
            qoe_list, video_list, rebuf_list, smooth_list, _, _ = parse_result(result.strip())
            
            # Optimize using only the QoE values
            if len(qoe_list)>1:
                qoe = np.mean(qoe_list[1:])
            else:
                qoe = -1000000
            
            # Other metrics can be saved for analysis
            metrics = {
                'qoe': qoe,
                'bitrate_mean': np.mean(video_list),
                'rebuf_mean': np.mean(rebuf_list),
                'smooth_mean': np.mean(smooth_list)
            }
            
            # If necessary, metrics can be saved to class attributes or external storage
            # self.optimization_metrics.append(metrics)
            
            return qoe
            
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            return float('-inf')
        except Exception as e:
            print(f"Unexpected error in objective function: {e}")
            return float('-inf')

    # Define the optimizer
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds={'discount_factor': (0, 5)},
        random_state=42
    )

    # Set the initial point
    optimizer.probe(
        params={'discount_factor': discount_factor},
        lazy=True
    )

    # Run the optimization
    print(f"Initial discount factor: {discount_factor}")
    print(f"Starting optimization...\n")

    optimizer.maximize(
        init_points=5,    # Number of points for random exploration
        n_iter=15,        # Number of iterations for Bayesian optimization
    )

    # Print the optimization process
    print("\nOptimization process:")
    for i, res in enumerate(optimizer.res):
        print(f"Iteration {i+1}: discount_factor = {res['params']['discount_factor']:.4f}, QoE = {res['target']:.4f}")

    # Print the optimal results
    print(f"\nBest result:")
    print(f"Optimal discount factor: {optimizer.max['params']['discount_factor']:.4f}")
    print(f"Optimal QoE: {optimizer.max['target']:.4f}")

    last_three = sorted(optimizer.res[-3:], key=lambda x: x['target'], reverse=True)
    return optimizer.max['params']['discount_factor']
if __name__ == "__main__":
    test_bw = [31.869401,46.722848,28.085088,56.85248,52.317506,46.225361,42.549866,28.298058,37.907437,55.195996,35.05472,27.644064,24.603712,23.16464,8.00048,9.554624,15.183872,1.885315,42.391159,32.934336,30.545039,33.17488,27.008352,29.332064,23.840735,22.441504,17.069216,11.898138,16.617728,12.062528,20.516987,14.871136,2.979363,0.010784,2.33779,6.282571,11.409472,9.976008,9.414432,12.418816,9.343296,5.753626,0.948044,11.05387,1.747008,33.064608,34.797056,37.84171,47.144521,35.935744,26.258624,23.264575,20.697241,21.548,14.608496,16.54179,16.77056,14.85628,14.304768,18.909952,20.220416,19.505778,16.905686,17.047672] # Your bandwidth sequence
    test_time = [0.431,1.431,2.431,3.431,4.432,5.431,6.432,7.431,8.432,9.431,10.431,11.431,12.431,13.431,14.431,15.431,16.431,17.432,18.431,19.431,20.432,21.432,22.431,23.431,24.432,25.432,26.432,27.431,28.431,29.431,30.432,31.432,32.431,33.431,34.432,35.431,36.431,37.432,38.432,39.432,40.432,41.431,42.432,43.431,44.431,45.431,46.431,47.432,48.431,49.431,50.431,51.432,52.431,53.431,54.432,55.431,56.431,57.432,58.432,59.432,60.432,61.431,62.432,63.431] # Your time sequence
    test_buffer = 0
    test_video = 0
    test_discount = 1

    # Run the tests
    last_three = bayes(test_bw, test_time, test_buffer, test_video, test_discount)

    # Print the results
    print("\nLast 3 iterations (possible best regions):")
    for i, res in enumerate(last_three):
        print(f"Rank {i+1}: discount_factor = {res['params']['discount_factor']:.4f}, QoE = {res['target']:.4f}")