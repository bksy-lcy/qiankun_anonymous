from typing import List, Dict, Any, NamedTuple
import numpy as np
from dataclasses import dataclass
import json
from scipy import signal, stats
from copy import deepcopy
from opt_new import ABROptimizer
from llm_manager import get_llm_manager
@dataclass
class EvaluationResult:
    continue_test: str
    need_improver: str
    loop_done: str
    confidence: float
    conclusion: str

class MetricsData(NamedTuple):
    qoe: List[float]
    bitrate: List[float]
    rebuffer: List[float]
    smoothness: List[float]

class LLMAgent:
    def __init__(self, min_samples: int = 20):
        self.min_samples = min_samples
        self.current_metrics = {}  # Dict[str, MetricsData]
        self.network_trace = None
        self.time_trace = None
        self.badcase_bw = []
        self.badcase_time = []
        self.badcase_metrics = {}  # Dict[str, MetricsData]
        self.history_bw = []
        self.history_metrics = {}  # Dict[str, MetricsData]
        self.parameter = 1
        self.alg_test = None
        
        # self.LLM = LLM.BandwidthPredictor()
    def reset(self, alg_test,dataset,file_name):
        """Reset agent state"""
        self.current_metrics = {}
        self.network_trace = None
        self.time_trace = None
        self.badcase_bw = []
        self.badcase_time = []
        self.badcase_metrics = {}
        self.history_bw = []
        self.history_metrics = {}
        self.parameter = 1
        self.alg_test = alg_test
        self.optimizer = ABROptimizer(dataset,file_name)

    def observe(self, network_trace: List[float], time_trace: List[float], 
                metrics: Dict[str, Dict[str, List[float]]]) -> None:
        """
        Update Observation Data
        
        Args:
            network_trace: bandwidth data
            metrics: dictionary containing all metrics for each algorithm
                Format: {
                    'algo_name': {
                        'qoe': [...],
                        'bitrate': [...],
                        'rebuf': [...],
                        'smooth': [...]
                    }
                }
        """
        print('save',network_trace,time_trace)
        self.network_trace = network_trace
        self.time_trace = time_trace
        self.current_metrics = {
            algo: MetricsData(
                qoe=data['qoe'],
                bitrate=data['bitrate'],
                rebuffer=data['rebuf'],
                smoothness=data['smooth']
            )
            for algo, data in metrics.items()
        }

    def evaluator(self, current_bw: List[float], current_time: List[float], 
                 current_parameter: float) -> EvaluationResult:
        """Evaluate current performance and determine next steps"""
        try:
            # Check if there is enough data for optimization
            has_sufficient_data = self._check_data_sufficiency(
                self.current_metrics,
                self.history_metrics,
            )
            print('has_sufficient_data ',has_sufficient_data )
            self.update_history(current_bw)
            if not has_sufficient_data:
                return EvaluationResult(
                    continue_test='Y',
                    need_improver='N',
                    loop_done='N',
                    confidence=0.0,
                    conclusion='Insufficient data for optimization'
                )
            print('has updated')
            # Analyzing Performance - Algorithm Comparison Using QoE Only
            current_qoe = {algo: metrics.qoe 
                          for algo, metrics in self.current_metrics.items()}
            prompt = self._construct_prompt_eva(current_qoe)
            LLM = get_llm_manager()
            algorithm, confidence, decision = LLM.evaluate_algorithms(
                prompt, 
                list(self.current_metrics.keys())
            )
            
            # Update History
            self.parameter = current_parameter
            
            need_improver = 'Y' if self.alg_test not in algorithm   else 'N'
            loop_done = 'Y' if need_improver == 'N' and decision == 'N' else 'N'

            return EvaluationResult(
                continue_test=decision,
                need_improver=need_improver,
                loop_done=loop_done,
                conclusion=algorithm,
                confidence=confidence
            )

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return EvaluationResult(
                continue_test='Y',
                need_improver='N',
                loop_done='N',
                conclusion='Evaluation failed',
                confidence=0.0
            )

    def save_badcase(self, bw_list: List[float], time_list: List[float]) -> None:
        """Save Bad Cases"""
        self.badcase_bw.extend(bw_list)
        self.badcase_time.extend(time_list)
        self.badcase_metrics = self.merge_metrics_results(
            self.badcase_metrics, 
            self.current_metrics
        )

    def update_history(self, bw_list: List[float]) -> None:
        """Update historical data"""
        self.history_bw.extend(bw_list)
        # print(  self.history_metrics, 
        #     self.current_metrics)
        self.history_metrics = self.merge_metrics_results(
            self.history_metrics, 
            self.current_metrics
        )
    def _check_data_sufficiency(self, current_metrics, history_metrics) -> bool:
        """Check if data is sufficient to differentiate algorithm performance"""
        
        # Calculate statistical information for the current metrics
        current_stats = {}
        for algo, metrics in current_metrics.items():
            current_stats[algo] = {
                'qoe_mean': np.mean(metrics.qoe),
                'qoe_var': np.var(metrics.qoe),
                'bitrate_mean': np.mean(metrics.bitrate),
                'rebuffer_mean': np.mean(metrics.rebuffer),
                'smoothness_mean': np.mean(metrics.smoothness)
            }

        # Build the prompt
        prompt = f"""Assess if the following performance data is sufficient to differentiate between algorithms:

        Current metrics for each algorithm:"""

        # Add current metrics information
        for algo, stats in current_stats.items():
            prompt += f"""{algo}:
                    QoE: mean={stats['qoe_mean']:.2f}, variance={stats['qoe_var']:.2f}
                    Bitrate: mean={stats['bitrate_mean']:.2f}
                    Rebuffer: mean={stats['rebuffer_mean']:.2f}
                    Smoothness: mean={stats['smoothness_mean']:.2f}"""

        # If historical data is available, add historical metrics information
        if history_metrics:
            prompt += "\n\nHistorical metrics for each algorithm:"
            for algo, metrics in history_metrics.items():
                history_stats = {
                    'qoe_mean': np.mean(metrics.qoe),
                    'qoe_var': np.var(metrics.qoe),
                    'bitrate_mean': np.mean(metrics.bitrate),
                    'rebuffer_mean': np.mean(metrics.rebuffer),
                    'smoothness_mean': np.mean(metrics.smoothness)
                }
                prompt += f"""
                {algo}:
                    QoE: mean={history_stats['qoe_mean']:.2f}, variance={history_stats['qoe_var']:.2f}
                    Bitrate: mean={history_stats['bitrate_mean']:.2f}
                    Rebuffer: mean={history_stats['rebuffer_mean']:.2f}
                    Smoothness: mean={history_stats['smoothness_mean']:.2f}"""

        prompt += """

        Please evaluate if this data is sufficient to clearly differentiate the performance between algorithms, considering:
        1. The gap between different algorithms' QoE
        2. The stability of measurements (variance)
        3. The consistency between current and historical data (if available)
        4. The sample size and diversity
        IMPORTANT: Your response MUST follow this exact format:
        1. First line: ONLY the word "yes" or "no" (nothing else)
        2. Second line: Your explanation

        REMEMBER: Your response must be formatted exactly as:
        yes/no
        Your explanation here"""
        llm_manager = get_llm_manager()
        response = llm_manager.get_raw_response(prompt)
        first_line = response.split('\n')[0].strip().lower()
        return first_line == "yes"
    def merge_metrics_results(self, existing: Dict[str, MetricsData], 
                            new: Dict[str, MetricsData]) -> Dict[str, MetricsData]:
        """Merge Performance Metrics Results"""
        if not existing:
            return new
        merged = {}
        for alg in set(existing.keys()) | set(new.keys()):
            if alg not in existing:
                merged[alg] = new[alg]
            elif alg not in new:
                merged[alg] = existing[alg]
            else:
                merged[alg] = MetricsData(
                    qoe=list(existing[alg].qoe) + list(new[alg].qoe),
                    bitrate=list(existing[alg].bitrate) + list(new[alg].bitrate),
                    rebuffer=list(existing[alg].rebuffer) + list(new[alg].rebuffer),
                    smoothness=list(existing[alg].smoothness) + list(new[alg].smoothness)
                )
        return merged
    def get_stats(self, data_list):
        """
        Calculate basic statistics for a list of numerical data
        
        Args:
            data_list: List of numerical values
            
        Returns:
            Dictionary containing mean, std dev, min and max values
        """
        if not data_list:
            return {
                'mean': 0,
                'std': 0, 
                'min': 0,
                'max': 0
            }
            
        stats = {
            'mean': np.mean(data_list),
            'std': np.std(data_list),
            'min': min(data_list),
            'max': max(data_list)
        }
        
        return stats
    def save_bw(self,bw,time):
        self.optimizer.save_bw(bw,time)
    def _construct_prompt_eva(self, qoe_results: Dict[str, List[float]]) -> str:
        """Construct an evaluation prompt with detailed information on performance metrics"""
        output = "Compare these algorithms:\n"
        
        for alg, metrics in self.current_metrics.items():
            stats_qoe = self.get_stats(metrics.qoe)
            stats_bitrate = self.get_stats(metrics.bitrate)
            stats_rebuffer = self.get_stats(metrics.rebuffer)
            stats_smoothness = self.get_stats(metrics.smoothness)
            
            output += f"\nAlgorithm {alg} statistics:\n"
            output += f"Score Metrics:\n"
            output += f"- Mean: {stats_qoe['mean']:.2f} ± {stats_qoe['std']:.2f}\n"
            output += f"- Range: [{stats_qoe['min']:.2f}, {stats_qoe['max']:.2f}]\n"
            
            output += f"Sample size: {len(metrics.qoe)}\n\n"

        output += "Based on Score your response MUST follow this structure:\n"
        output += "Best algorithm:\n"
        output += "Confidence level:(0-1)\n"
        output += "Whether need more data:Y/N\n"
        output += "Explanation:Y/N\n"
        
        return output
    def predict(self,bw,duration_series, timestamp_series):
        LLM = get_llm_manager()
        return LLM.predict_bandwidth(bw, duration_series, timestamp_series)
    def opt_normal(self):
        self.optimizer.opt_normal()
    def optimize_parameter(self) -> float:
        """
        Optimization Algorithm Parameters
        
        Returns:
            float: the optimized parameter, or the current parameter if no optimization is needed
        """
        try:
         
            print('start agent opt')
            # call optimizer
            result,normal_params = self.optimizer.process_badcase(
                new_badcase_bw=self.network_trace,
                new_badcase_time=self.time_trace,
                param = self.parameter
            )
               
            
            print('result:',result)
            if result['status'] == 'success':
                # Update parameters
                new_parameter = result['params']
       
                return new_parameter,normal_params 
                
            # If no optimization is needed or there is insufficient data, keep the current parameters
            return self.parameter,normal_params 

        except Exception as e:
            print(f"Parameter optimization failed: {e}")
            return self.parameter,normal_params 
    def save_new_qoe(self,current_qoe, new_parameter):
        self.parameter = new_parameter

def generate_sample_data(size: int = 64) -> Dict:
    """Generate test data"""
    # Generate bandwidth sequences
    bw = np.random.normal(1000, 200, size)  # A bandwidth sequence with a mean of 1000 Kbps and a standard deviation of 200
    bw = np.clip(bw, 100, 2000)  # Limited to the range of 100-2000Kbps
    
    # Generate time sequences
    time = np.arange(size) * 1.0  # 1-second interval
    
    # Generate performance metrics
    metrics = {
        'bb': {
            'qoe': list(np.random.normal(100, 20, size)),
            'bitrate': list(np.random.normal(1500, 300, size)),
            'rebuf': list(np.abs(np.random.normal(0.1, 0.05, size))),
            'smooth': list(np.abs(np.random.normal(0.2, 0.1, size)))
        },
        'mpc': {
            'qoe': list(np.random.normal(110, 20, size)),
            'bitrate': list(np.random.normal(1600, 300, size)),
            'rebuf': list(np.abs(np.random.normal(0.08, 0.04, size))),
            'smooth': list(np.abs(np.random.normal(0.15, 0.08, size)))
        },
        'm_pc': {
            'qoe': list(np.random.normal(105, 20, size)),
            'bitrate': list(np.random.normal(1550, 300, size)),
            'rebuf': list(np.abs(np.random.normal(0.09, 0.04, size))),
            'smooth': list(np.abs(np.random.normal(0.18, 0.09, size)))
        }
    }
    
    return {
        'bw': list(bw),
        'time': list(time),
        'metrics': metrics
    }

def main():
    # Initialized LLMAgent
    agent = LLMAgent(min_samples=20)
    print("1. Initialized LLMAgent")
    
    # Test: Reset Function
    agent.reset('m_pc')
    print("2. Reset agent with test algorithm 'm_pc'")
    
    # Generate sample data
    sample_data = generate_sample_data()
    print("3. Generated sample data")
    
    # Test: Observe Function
    agent.observe(sample_data['bw'], sample_data['metrics'])
    print("4. Updated agent with observation data")
    print(f"   Network trace size: {len(agent.network_trace)}")
    print(f"   Current metrics keys: {list(agent.current_metrics.keys())}")
    
    # Test:  Evaluation Function
    evaluation_result = agent.evaluator(
        sample_data['bw'], 
        sample_data['time'],
        0.9  # initial parameter
    )
    print("\n5. Evaluation result:")
    print(f"   Continue test: {evaluation_result.continue_test}")
    print(f"   Need improver: {evaluation_result.need_improver}")
    print(f"   Loop done: {evaluation_result.loop_done}")
    print(f"   Confidence: {evaluation_result.confidence}")
    print(f"   Conclusion: {evaluation_result.conclusion}")
    
    # Test: save badcase
    if evaluation_result.need_improver == 'Y':
        agent.save_badcase(sample_data['bw'], sample_data['time'])
        print("\n6. Saved badcase")
        print(f"   Badcase bandwidth sequences: {len(agent.badcase_bw)}")
        print(f"   Badcase metrics: {list(agent.badcase_metrics.keys())}")
    
    # Test: history update
    agent.update_history(sample_data['bw'])
    print("\n7. Updated history")
    print(f"   History bandwidth sequences: {len(agent.history_bw)}")
    print(f"   History metrics: {list(agent.history_metrics.keys())}")
    print('current parameter',agent.parameter)
    # Test: parameter optimization
    new_parameter = agent.optimize_parameter()
    print("\n8. Parameter optimization result:")
    print(f"   New parameter: {new_parameter}")
    
    # Test: Bandwidth Prediction
    predicted_bw = agent.predict(
        sample_data['bw'],
        sample_data['time'],
        sample_data['time']
    )
    print("\n9. Bandwidth prediction:")
    print(f"   Predicted bandwidth shape: {predicted_bw.shape}")
    
    # Test: get optimization summary
    if hasattr(agent, 'get_optimization_summary'):
        summary = agent.get_optimization_summary()
        print("\n10. Optimization summary:")
        print(f"   Current parameter: {summary['current_parameter']}")
        print(f"   Current distribution: {summary['current_distribution']}")

def test_error_handling():
    """Test: Error Handling"""
    agent = LLMAgent()
    
    # Test: empty data processing
    try:
        agent.observe([], {})
        print("Empty data handled successfully")
    except Exception as e:
        print(f"Error handling empty data: {e}")
        
    # Test: format error data handling
    try:
        agent.observe([1,2,3], {'invalid_format': [1,2,3]})
        print("Invalid format handled successfully")
    except Exception as e:
        print(f"Error handling invalid format: {e}")

if __name__ == "__main__":
    print("=== Starting main test ===")
    main()
    print("\n=== Starting error handling test ===")
    test_error_handling()
    print("\n=== All tests completed ===")