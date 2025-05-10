from typing import List, Dict, Tuple, Optional
import numpy as np
import os
import subprocess
import LLMagent_new as LLMAgent
from dataclasses import dataclass

@dataclass
class AlgorithmState:
    buffer: float
    video_id: int

class ExperimentManager:
    def __init__(self, agent: LLMAgent, algorithms: List[str]):
        self.agent = agent
        self.algorithms = algorithms
        self.results = {algo: [] for algo in algorithms}
        self.states = {algo: AlgorithmState(0, 0) for algo in algorithms}
        self.previous_states = {algo: AlgorithmState(0, 0) for algo in algorithms}
        
        self.test_alg = 'm_pc'
        self.parameter = 1
        self.need_improver = False
        self.continue_test = True

    def run_abr(self, bw: List[float], time: List[float], 
                algo: str, state: AlgorithmState) -> Tuple[List[float], List[float], List[float], List[float], float, int]:
        """
        Run ABR algorithm and return results
        
        Returns:
            Tuple containing:
            - QoE rewards list
            - Video bitrate selections list
            - Rebuffering events list
            - Smoothness penalties list
            - Current buffer size
            - Video chunk position
        """
        bw_str = str(bw).replace(' ', '')
        time_str = str(time).replace(' ', '')
        
        cmd = f'python {algo}.py {bw_str} {time_str} {state.buffer} {state.video_id}'
        if algo == self.test_alg:
            cmd += f' {self.parameter}'
            
        try:
            result = subprocess.check_output(cmd, shell=True, text=True)
            return self._parse_abr_result(result.strip())
        except subprocess.CalledProcessError as e:
            print(f"ABR execution failed: {e}")
            return [], [], [], [], 0, 0
    def run_abr_bayes(self, bw: List[float], time: List[float],dataset,file_name) -> Tuple[List[float], List[float], List[float], List[float], float, int]:
        """
        Run ABR algorithm and return results
        
        Returns:
            Tuple containing:
            - QoE rewards list
            - Video bitrate selections list
            - Rebuffering events list
            - Smoothness penalties list
            - Current buffer size
            - Video chunk position
        """
        bw_str = str(bw).replace(' ', '')
        time_str = str(time).replace(' ', '')
        cmd = f'python bayes_pc.py {bw_str} {time_str} {0} {0} 3 {dataset} {file_name}'


           
            
        print(cmd)
        os.system(cmd)
    def _parse_abr_result(self, result_str: str) -> Tuple[List[float], List[float], List[float], List[float], float, int]:
        """
        Parsing the ABR algorithm output
        
        Returns:
            Tuple containing:
            - r_batch: QoE rewards list
            - video_batch: Video bitrate selections list
            - rebuf_batch: Rebuffering events list
            - smooth_batch: Smoothness penalties list
            - buffer_size: Current buffer size
            - remaining_chunks: Remaining video chunks (48 - video_chunk_remain)
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

            if len(brackets) != 4:  # There should be 4 lists
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

            return (r_batch, video_batch, rebuf_batch, smooth_batch, 
                    buffer_size, int(chunks_remain))

        except Exception as e:
            print(f"Result parsing failed: {e}")
            return [], [], [], [], 0, 0

    def optimize_parameters(self, bw: List[float], time: List[float]) -> None:
        """Optimizate algorithm parameters"""
        last_qoe = self.results[self.test_alg][-1]
        last_parameter = self.parameter
        current_qoe = float('-inf')
        print('start opt',current_qoe , last_qoe,last_parameter )
        opt_cnt = 0
        # Get new parameters
        opt_cnt += 1
        new_parameter,normal_params = self.agent.optimize_parameter()
            
        self.parameter = new_parameter
        print(f'Trying parameter: {self.parameter}')
        
        # Test new parameters
        state = self.previous_states[self.test_alg]
        qoe_list,video_batch, rebuf_batch, smooth_batch, buffer, video_id  = self.run_abr(bw, time, self.test_alg, state)
        current_qoe = np.mean(qoe_list)
            
        # Updated Record
        if current_qoe > last_qoe:
            print(f'Performance: current QoE = {current_qoe}, last QoE = {last_qoe}')
        self.parameter = normal_params

    def process_bandwidth_data(self, bw: List[float], time: List[float]) -> None:
        metrics = {algo: {
            'qoe': [],
            'bitrate': [],
            'rebuf': [],
            'smooth': []
        } for algo in self.algorithms}
        
        for algo in self.algorithms:
            r_batch, video_batch, rebuf_batch, smooth_batch, buffer, video_id = \
                self.run_abr(bw, time, algo, self.states[algo])
            if len(r_batch) == 0:
                r_batch = [0]
            metrics[algo]['qoe'].extend(r_batch)
            metrics[algo]['bitrate'].extend(video_batch)
            metrics[algo]['rebuf'].extend(rebuf_batch)
            metrics[algo]['smooth'].extend(smooth_batch)
            
            self.states[algo] = AlgorithmState(buffer, video_id)
            self.results[algo].append(np.mean(r_batch))
        # Calculate the average QoE for each algorithm in this round
        avg_qoes = {algo: np.mean(metrics[algo]['qoe'][-len(r_batch):]) for algo in self.algorithms}
        best_algo = max(avg_qoes, key=avg_qoes.get)

        # If the highest QoE is not the test algorithm, save the badcase
        if best_algo != self.test_alg:
            self.agent.save_badcase(bw, time)
        self.agent.observe(bw, time,metrics)
    def check_experiment_status(self, bw: List[float], time: List[float]) -> Tuple[bool, bool]:
        """Checks the status of the experiment and returns flags for continued testing and needed improvements"""
        decision = self.agent.evaluator(bw, time, self.parameter)
        print('decision:',decision)
        return (
            decision.continue_test == 'Y',
            decision.need_improver == 'Y'
        )

    def run_experiment(self, bandwidth_series: List[float], 
                      duration_series: List[float], 
                      timestamp_series: List[float],dataset,file_name ) -> Optional[float]:
        """Run the full experiment"""
        self.agent.reset(self.test_alg,dataset,file_name )
        
        for epoch in range(100):
            start_idx = epoch * 25
            if start_idx >= 0.5 *len(bandwidth_series):
                print('Bandwidth data exhausted')
                self.agent.opt_normal()
                return None
                
            # Get current batch data
            current_bw = bandwidth_series[start_idx:start_idx + 25]
            current_time = timestamp_series[start_idx:start_idx + 25]
            current_dur = duration_series[start_idx:start_idx + 25]
            # Process real bandwidth data
            print('real bw',current_bw,current_time)
            self.process_bandwidth_data(current_bw, current_time)
            self.continue_test, self.need_improver = self.check_experiment_status(
                current_bw, current_time
            )  
            if self.need_improver:
                self.optimize_parameters(current_bw, current_time)
            else:
                self.agent.save_bw(current_bw,current_time)
            print('length:' ,len(self.agent.history_bw))
            # Process predicted bandwidth data
            predicted_bw = self.agent.predict(current_bw, current_time, current_dur)
            self.process_bandwidth_data(predicted_bw, current_time)
            self.continue_test, self.need_improver = self.check_experiment_status(
                predicted_bw, current_time
            )
            if self.need_improver:
                self.optimize_parameters(predicted_bw, current_time)
        print('Maximum iterations reached without conclusion')
        self.agent.opt_normal()
        return self.parameter
def load_trace(cooked_trace_folder):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_dur = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        cooked_dur = []
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
                cooked_dur.append(float(parse[0])*1000)
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)
        all_cooked_dur.append(cooked_dur)

    return all_cooked_bw, all_cooked_dur, all_cooked_time,all_file_names

def main():
    # init
    agent =LLMAgent.LLMAgent()
    algorithms = ['bb', 'mpc', 'rl_no_training', 'm_pc']
    manager = ExperimentManager(agent, algorithms)
    import sys
    dataset = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    # load data
    dir_path = '../Design/test/'+dataset
    bw_list, dur_list, time_list, file_names = load_trace(dir_path)
    if not os.path.exists('bayes_rules/'+dataset):
        os.mkdir('bayes_rules/'+dataset)
    # run experiment
    flag = 0
    for bw, dur, time, file_name in zip(bw_list, dur_list, time_list, file_names):
        result = manager.run_experiment(bw, dur, time,dataset,file_name )
        print(f'Result for {file_name}: {result}')
        

if __name__ == "__main__":
    main()