import os
from multiprocessing import Pool
from itertools import product

def run_command(args):
    dataset = args
    cmd = f"python bayes_mpc.py {dataset}"
    print(f"Running: {cmd}")
    return os.system(cmd)

if __name__ == '__main__':
    # Get the number of CPU cores
    cpu_count = os.cpu_count()
    # Set the number of processes, it is recommended to set the number of CPU cores or the number of cores -1
    process_count = cpu_count - 1 if cpu_count > 1 else 1
    
    # Create all parameter combinations
    parameter_combinations = os.listdir('results/')
    
    print(f"Total tasks: {len(parameter_combinations)}")
    print(f"Using {process_count} processes")
    
    pool = None
    try:
        # Creating Process Pool
        pool = Pool(processes=process_count)
        # Use map to perform task and get the result
        results = pool.map(run_command, parameter_combinations)
            
        # Checking results
        failed_tasks = sum(1 for r in results if r != 0)
        if failed_tasks > 0:
            print(f"Warning: {failed_tasks} tasks failed")
        else:
            print("All tasks completed successfully")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        if pool:
            pool.terminate()
            pool.join()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if pool:
            pool.terminate()
            pool.join()
    finally:
        if pool:
            pool.close()
            pool.join()