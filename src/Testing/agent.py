from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
import json
import numpy as np
from scipy import signal
import sys
sys.path.append("..") 
import llm_model.LLM as LLM
import subprocess
from pathlib import Path

# Constants
MIN_SAMPLES = 20
BATCH_SIZE = 25
MAX_EPOCHS = 100
METRICS = ['qoe', 'bitrate', 'rebuf', 'smooth']

@dataclass
class EvaluationResult:
    decision: str  # 'Y' or 'N'
    conclusion: str
    confidence: float

class Agent(ABC):
    @abstractmethod
    def observe(self, state: Any) -> None:
        pass
    
    @abstractmethod
    def think(self) -> Any:
        pass
    
    @abstractmethod
    def act(self) -> Any:
        pass

class MetricsAnalyzer:
    @staticmethod
    def get_stats(scores: List[float]) -> Dict[str, float]:
        scores_array = np.array(scores)
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'p5': float(np.percentile(scores_array, 5)),
            'p95': float(np.percentile(scores_array, 95)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array))
        }

class LLMEvaluator(Agent):
    def __init__(self, min_samples: int = MIN_SAMPLES):
        self.min_samples = min_samples
        self.llm = LLM.BandwidthPredictor()
        # self.reset()

    def reset(self,bandwidth_series_total_len) -> None:
        self.evaluation_data = {}
        self.last_evaluation = None
        self.bandwidth_series_total_len = bandwidth_series_total_len

    def observe(self, evaluation_data: Dict[str, Dict]) -> None:
        """
        Receive evaluation data
        evaluation_data contains:
        - historical_metrics: historical real data
        - current_batch: current batch real data
        - predicted_metrics: historical predicted data
        - predicted_batch: current batch predicted data
        - bandwidth_data: bandwidth-related data
        """
        self.evaluation_data = evaluation_data

    def think(self) -> EvaluationResult:
        try:
            algorithms = list(self.evaluation_data['predicted_metrics'].keys())
            prompt = self._construct_prompt()

            algorithm, confidence, decision = self.llm.LLMout(prompt, algorithms)
            print(algorithm, confidence, decision )
         
            if not all([algorithm, confidence, decision]):
                return EvaluationResult('Y', 'LLM output failed', 0.0)
            
            evaluation = EvaluationResult(decision, algorithm, confidence)
            self.last_evaluation = evaluation
            return evaluation
            
        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            return EvaluationResult('Y', 'LLM evaluation failed', 0.0)

    def act(self) -> str:
        return self.think().decision

    def predict(self, bw: List[float], duration_series: List[float], 
                timestamp_series: List[float]) -> np.ndarray:
        return self.llm.predict(bw, duration_series, timestamp_series)

    def _construct_prompt(self) -> str:
        bandwidth_series_total_len = self.bandwidth_series_total_len
        output = ["Compare these ABR algorithms based on both real and predicted data:"]
    
        # Add data sampling information
        bandwidth_data = self.evaluation_data['bandwidth_data']
        current_bw = bandwidth_data['current_bandwidth']
        current_samples = len(current_bw)
        historical_samples = len(bandwidth_data['historical_bandwidth']) 
        
        # Calculating Sampling Progress
        current_progress = (current_samples/bandwidth_series_total_len*100)
        is_initial_sampling = (historical_samples == 0)
        
        output.extend([
            "\nData Sampling Progress:",
            f"Total available samples: {bandwidth_series_total_len}",
            f"Current batch samples: {current_samples} ({current_progress:.1f}% of total)",
        ])
 
        output.append("\nBandwidth Analysis:")

        # current bandwidth
        output.extend([
            "\nCurrent Bandwidth:",
            f"Mean: {np.mean(current_bw):.2f} Mbps",
            f"Range: {min(current_bw):.2f} - {max(current_bw):.2f} Mbps"
        ])
        
        # Forecasted bandwidth statistics
        if 'predicted' in bandwidth_data['bandwidth_stats']:
            stats = bandwidth_data['bandwidth_stats']['predicted']
            output.extend([
                "\nPredicted Bandwidth Statistics:",
                f"Mean: {stats['mean']:.2f} Mbps",
                f"Standard Deviation: {stats['std']:.2f}",
                f"Range: {stats['min']:.2f} - {stats['max']:.2f} Mbps",
                f"Variation Coefficient: {stats['variation']:.2f}"
            ])
        
        # Bandwidth trend analysis
        output.extend([
            "\nBandwidth Trend Analysis:",
            self._analyze_bandwidth_trend(bandwidth_data)
        ])
        

        output.append("\nPerformance Statistics:")
        self._add_metrics_stats(self.evaluation_data['predicted_metrics'], output)
    
        output.extend([
            "\nIMPORTANT: You MUST use the exact format below for your response.",
            "\nRequired Response Format:",
            "---",
            "Best algorithm: [first algorithm name]",
            "Confidence level: [number between 0 and 1]",
            "Whether need more data: [Y or N only]",
            "Explanation: [your detailed analysis including:]",
            "- Whether current patterns can predict full dataset performance",
            "- How well current samples represent overall network conditions",
            "- Reliability of performance trend extrapolation",
            "- If Y, what additional patterns needed to predict full dataset behavior",
            "- If N, why current patterns sufficient to predict overall performance",
            "---",
            "\nNote:",
            "- Algorithm name must be one of the evaluated algorithms",
            "- Confidence level must be a decimal number between 0 and 1",
           
        ])
        
     
        
        return "\n".join(output)

    def _add_metrics_stats(self, metrics: Dict, output: List[str]) -> None:
        for algo in metrics:
            output.append(f"\nAlgorithm {algo}:")
            for metric in METRICS:
                if metrics[algo][metric]:  # Only process if there's data
                    data = np.array(metrics[algo][metric])
                    mean = np.mean(data)
                    std = np.std(data)
                    if metric == 'qoe':
                        output.append(f"{metric.upper()}: {mean:.2f} ± {std:.2f}")

    def _analyze_bandwidth_trend(self, bandwidth_data: Dict) -> str:
        """Analyze bandwidth trend"""
        historical = bandwidth_data['historical_bandwidth']
        current = bandwidth_data['current_bandwidth']
        predicted = bandwidth_data['predicted_bandwidth']
        
        if not historical:
            return "Insufficient historical data for trend analysis"
            
        # Calculate trend
        recent_mean = np.mean(current)
        hist_mean = np.mean(historical)
        pred_mean = np.mean(predicted)
        
        # Calculate the rate of change
        bandwidth_trend = (recent_mean - hist_mean) / hist_mean * 100
        future_trend = (pred_mean - recent_mean) / recent_mean * 100
        
        # Build trend descriptions
        trend_description = [
            f"Recent trend: {'increasing' if bandwidth_trend > 0 else 'decreasing'} by {abs(bandwidth_trend):.1f}%",
            f"Predicted trend: {'increasing' if future_trend > 0 else 'decreasing'} by {abs(future_trend):.1f}%",
            f"Bandwidth stability: {'stable' if abs(bandwidth_trend) < 10 else 'volatile'}"
        ]
        
        return "\n".join(trend_description)
class ExperimentManager:
    def __init__(self, evaluator: LLMEvaluator, algorithms: List[str]):
        self.evaluator = evaluator
        self.algorithms = algorithms
        # self.reset()

    def reset(self) -> None:
        self.real_metrics = {algo: {metric: [] for metric in METRICS} 
                           for algo in self.algorithms}
        self.pred_metrics = {algo: {metric: [] for metric in METRICS} 
                           for algo in self.algorithms}
        self.abr_state = {algo: (0, 0) for algo in self.algorithms}
        # Add bandwidth history
        self.bandwidth_history = []
        self.predicted_bandwidth_history = []

    @staticmethod
    def parse_abr_output(result: str) -> Tuple[List[float], ...]:
        try:
            # Find the location of all square brackets
            start_indices = []
            end_indices = []
            for i, char in enumerate(result):
                if char == '[':
                    start_indices.append(i)
                elif char == ']':
                    end_indices.append(i)

            if len(start_indices) != 4 or len(end_indices) != 4:
                raise ValueError(f"Expected 4 lists, found {len(start_indices)}")

            # Parsing four lists
            lists = []
            for start, end in zip(start_indices, end_indices):
                content = result[start+1:end]
                numbers = [float(x.strip()) for x in content.split(',') if x.strip()]
                lists.append(numbers)

            # Parsing the last buffer_size and chunks_remain
            remaining = result[end_indices[-1]+1:].strip()
            buffer_size, chunks_remain = map(float, remaining.split())

            return (*lists, buffer_size, int(chunks_remain))

        except Exception as e:
            print(f"Result parsing failed: {e}")
            print(f"Raw result: {result}")
            return [], [], [], [], 0, 0

    def run_abr(self, bandwidth_series: List[float], time_series: List[float], 
                algo: str, buffer: float, video_id: int) -> Tuple:
        cmd = (f'python {algo}.py {str(bandwidth_series).replace(" ", "")} '
               f'{str(time_series).replace(" ", "")} {buffer} {video_id}')
        print(f"Running command: {cmd}")  # Debug information
        result = subprocess.check_output(cmd, shell=True, text=True)
        print(f"Raw output from {algo}: {result}")  # Debug information
        return self.parse_abr_output(result)

    def process_batch(self, bw: List[float], time: List[float], 
                     is_predicted: bool) -> Dict[str, Dict[str, List[float]]]:
        """Process a batch of bandwidth data and return the metrics for that batch"""
        batch_metrics = {algo: {metric: [] for metric in METRICS} 
                        for algo in self.algorithms}
        
        # Record bandwidth data
        if is_predicted:
            self.predicted_bandwidth_history.extend(bw)
        else:
            self.bandwidth_history.extend(bw)
        
        for algo in self.algorithms:
            buffer, video_id = self.abr_state[algo]
            results = self.run_abr(bw, time, algo, buffer, video_id)
            
            if not results[0]:  # Empty results
                continue
                
            r_batch, video_batch, rebuf_batch, smooth_batch, buffer, video_id = results
            
            # Update the indicators for the current batch
            batch_metrics[algo]['qoe'].extend(r_batch)
            batch_metrics[algo]['bitrate'].extend(video_batch)
            batch_metrics[algo]['rebuf'].extend(rebuf_batch)
            batch_metrics[algo]['smooth'].extend(smooth_batch)
            
            # Update Status
            self.abr_state[algo] = (buffer, video_id)
        
        return batch_metrics

    def _get_bandwidth_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate bandwidth statistics"""
        stats = {}
        
        if self.bandwidth_history:
            hist_array = np.array(self.bandwidth_history)
            stats['historical'] = {
                'mean': float(np.mean(hist_array)),
                'std': float(np.std(hist_array)),
                'min': float(np.min(hist_array)),
                'max': float(np.max(hist_array)),
                'variation': float(np.std(hist_array) / np.mean(hist_array))  # Coefficient of variation
            }
        
        if self.predicted_bandwidth_history:
            pred_array = np.array(self.predicted_bandwidth_history)
            stats['predicted'] = {
                'mean': float(np.mean(pred_array)),
                'std': float(np.std(pred_array)),
                'min': float(np.min(pred_array)),
                'max': float(np.max(pred_array)),
                'variation': float(np.std(pred_array) / np.mean(pred_array))
            }
            
        return stats

    def run_experiment(self, bandwidth_series: List[float], 
                      duration_series: List[float], 
                      timestamp_series: List[float]) -> str:
        self.evaluator.reset(len(bandwidth_series))
        self.reset()
        
        for epoch in range(MAX_EPOCHS):
            start_idx = epoch * BATCH_SIZE
            if start_idx + BATCH_SIZE > len(bandwidth_series):
                print('fail end')
                return self.evaluator.last_evaluation.conclusion if self.evaluator.last_evaluation else 'evaluation failed'
                
            # Get data for the current batch
            batch_bw = bandwidth_series[start_idx:start_idx + BATCH_SIZE]
            batch_time = timestamp_series[start_idx:start_idx + BATCH_SIZE]
            duration_serie = duration_series[start_idx:start_idx + BATCH_SIZE]
            
            # Acquire and process predictive bandwidth data
            predicted_bw = self.evaluator.predict(
                batch_bw, batch_time, duration_serie)
            bw = batch_bw + predicted_bw
            time = timestamp_series[start_idx:start_idx + len(bw)]
            pred_batch_metrics = self.process_batch(bw, time, True)
            
            # Prepare bandwidth analysis data
            bandwidth_data = {
                'historical_bandwidth': self.bandwidth_history[:-BATCH_SIZE] if len(self.bandwidth_history) > BATCH_SIZE else [],
                'current_bandwidth': batch_bw,
                'predicted_bandwidth': predicted_bw,
                'bandwidth_stats': self._get_bandwidth_stats()
            }
            
            # Evaluation results
            evaluation_data = {
                'predicted_metrics':pred_batch_metrics,
                'predicted_batch': pred_batch_metrics,
                'bandwidth_data': bandwidth_data
            }

            self.evaluator.observe(evaluation_data)
          
            decision = self.evaluator.act()
        
            if self.evaluator.last_evaluation:
                return self.evaluator.last_evaluation.conclusion
            return 'evaluation failed'
                
        return 'fail end'

def load_trace(trace_folder: str) -> Tuple[List, List, List, List]:
    results = [], [], [], []
    trace_path = Path(trace_folder)
    
    for file_path in trace_path.glob('*'):
        with open(file_path, 'rb') as f:
            times, bws = [], []
            for line in f:
                try:
                    t, bw = map(float, line.split())
                    times.append(t)
                    bws.append(bw)
                except Exception as e:
                    print(f"Error processing line in {file_path}: {e}")
                    continue
            
            if times and bws:  # Only add if we got valid data
                results[0].append(bws)
                results[1].append([t * 1000 for t in times])
                results[2].append(times)
                results[3].append(file_path.name)
    
    return results

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python agent_new.py <dataset>")
        sys.exit(1)
        
    dataset = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    evaluator = LLMEvaluator()
    algorithms = ['bb', 'mpc', 'bola', 'hyb','rl_no_training','rl_ghent']

    manager = ExperimentManager(evaluator, algorithms)

    trace_dir = '../Design/test/'+dataset +'/'
    trace_data = load_trace(trace_dir)
    
    for bw, duration, timestamp, filename in zip(*trace_data):
        print(f"\nProcessing trace file: {filename}")
        result = manager.run_experiment(bw, duration, timestamp)
        print(f'Result for {filename}: {result}')
    
    print('Experiment completed successfully')

if __name__ == "__main__":
    main()