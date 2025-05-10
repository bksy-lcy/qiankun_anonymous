import numpy as np
from typing import List, Dict, Tuple
import json
from scipy import stats
from llm_manager import get_llm_manager
import bayes
import os
from data import DataManager
class ABROptimizer:
    def __init__(self,dataset,file_name):
        self.data_manager = DataManager(dataset,file_name)
        self.current_params = {
            'buffer_init': 30,
            'video_init': 0,
            'discount_factor': 1
        }
    def save_bw(self,bw,time):
        self.data_manager.add_bw(bw,time)
    def process_badcase(self, new_badcase_bw: List[float], new_badcase_time: List[float],param:float) -> Dict:
        """Process the new bad case"""
        self.current_params['discount_factor'] = param
        # Analyze the distribution and optimize the parameters
        results = self._analyze_and_handle_distribution(new_badcase_bw,new_badcase_time)
        self.save_current_state()
        
        return results,self.data_manager.data['normal_params'] 
            
    def _check_data_sufficiency(self, badcase_bw: List[float]) -> bool:
        """Check whether the data is sufficient for analysis"""
        if len(badcase_bw) < 10:  # Basic sample size check
            return False
                
        # Obtain the statistical characteristics of the current data
        current_stats = self._calculate_stats(badcase_bw)
        
        # Obtain historical normal data
        data = self.data_manager.get_all_data()
        normal_bw = data['normal_bw']
        
        # If there is no historical normal data, use basic rules for assessment
        if len(normal_bw) == 0:
            # Utilize basic rules: sample size, coefficient of variation, etc
            return (len(badcase_bw) >= 20 and 
                    current_stats['cv'] < 1.0)  # Basic stability requirements
        
        # Calculate the statistical characteristics of historical normal data
        normal_stats = self._calculate_stats(normal_bw)
        
        # Calculate the degree of variation in key indicators
        differences = {
            'mean_diff': abs(current_stats['mean'] - normal_stats['mean']) / normal_stats['mean'],
            'cv_diff': abs(current_stats['cv'] - normal_stats['cv']) / normal_stats['cv'],
            'skew_diff': abs(current_stats['skew'] - normal_stats['skew']),
            'peak_ratio_diff': abs(current_stats['peak_mean_ratio'] - normal_stats['peak_mean_ratio']) / normal_stats['peak_mean_ratio']
        }
        
        prompt = f"""Analysis of bandwidth data sufficiency:

            Current Data Statistics:
            - Sample size: {len(badcase_bw)}
            - Mean: {current_stats['mean']:.2f}
            - Standard deviation: {current_stats['std']:.2f}
            - CV: {current_stats['cv']:.2f}
            - Skewness: {current_stats['skew']:.2f}
            - Peak/Mean ratio: {current_stats['peak_mean_ratio']:.2f}

            Normal Pattern Statistics:
            - Sample size: {len(normal_bw)}
            - Mean: {normal_stats['mean']:.2f}
            - Standard deviation: {normal_stats['std']:.2f}
            - CV: {normal_stats['cv']:.2f}
            - Skewness: {normal_stats['skew']:.2f}
            - Peak/Mean ratio: {normal_stats['peak_mean_ratio']:.2f}

            Differences from Normal Pattern:
            - Mean difference: {differences['mean_diff']:.2%}
            - CV difference: {differences['cv_diff']:.2%}
            - Skewness difference: {differences['skew_diff']:.2f}
            - Peak/Mean ratio difference: {differences['peak_ratio_diff']:.2%}

            Based on these statistics, determine if this data is sufficient to:
            1. Reliably distinguish from normal distribution pattern
            2. Current sample size adequacy for statistical significance

            Your answer MUST ONLY "yes" or "no" Do not add any additional text.
            """
        
        # If the differences are significant, results can be returned directly without the need to invoke a LLM
        if (differences['mean_diff'] > 0.5 or  # The mean difference exceeds 50%
            differences['cv_diff'] > 0.5 or    # The coefficient of variation difference exceeds 50%
            differences['skew_diff'] > 1.0 or  # The difference in skewness is significant
            differences['peak_ratio_diff'] > 0.5):  # The difference in kurtosis exceeds 50%
            return True
            
        # In cases where the differences are not evident, LLM should be used for assessment
        llm_manager = get_llm_manager()
        response = llm_manager.get_raw_response(prompt).strip().lower()
        if "yes" in response:
            return True
        else:
            return False
    def opt_normal(self):
        data = self.data_manager.get_all_data()
        bw_all = data['normal_bw']
        time_all = data['normal_time']
        
        optimal_params = bayes.bayes(
            bw_all,
            time_all,
            buffer_init=0,
            video_init=0,
            discount_factor=self.data_manager.data['normal_params'] 
        )
        print(self.data_manager.data['normal_params'],optimal_params)
        self.data_manager.data['normal_params'] = optimal_params
        self.save_current_state()
    def _is_normal_distribution(self, data: List[float], normal_stats: Dict) -> Tuple[bool, float]:
        """Quickly check for normal distribution
        
        Returns:
            Tuple[bool, float]: (Is it normally distributed, confidence level)
        """
        current_stats = self._calculate_stats(data)
        
        # Calculate the deviation of key metrics from the normal distribution
        mean_diff = abs(current_stats['mean'] - normal_stats['mean']) / normal_stats['mean']
        cv_diff = abs(current_stats['cv'] - normal_stats['cv']) / normal_stats['cv']
        skew_diff = abs(current_stats['skew'] - normal_stats['skew'])
        
        # Set Thresholds
        THRESHOLDS = {
            'mean': 0.2,   # Mean deviation should not exceed 20%
            'cv': 0.3,     # Coefficient of variation deviation should not exceed 30%
            'skew': 0.5    # Skewness deviation should not exceed 0.5
        }
        
        # Calculate the confidence level
        confidence = 1.0
        confidence *= max(0, 1 - mean_diff/THRESHOLDS['mean'])
        confidence *= max(0, 1 - cv_diff/THRESHOLDS['cv'])
        confidence *= max(0, 1 - skew_diff/THRESHOLDS['skew'])
        
        # If all metrics are within the thresholds, it is considered to be normally distributed
        is_normal = (mean_diff <= THRESHOLDS['mean'] and 
                    cv_diff <= THRESHOLDS['cv'] and 
                    skew_diff <= THRESHOLDS['skew'])
        
        return is_normal, confidence
    def _analyze_and_handle_distribution(self, badcase_bw: List[float],badcase_time: List[float]) -> Dict:
        """Utilize a LLM to analyze the type of bandwidth distribution and process the results."""
        data = self.data_manager.get_all_data()
        normal_bw = data['normal_bw']
        normal_time = data['normal_time']
        anomaly_patterns = data['anomaly_patterns']
        print(normal_bw,badcase_bw,self.current_params['discount_factor'])
        # If there is no normally distributed data, use the current data as the baseline for normal distribution
        if len(normal_bw) == 0:
            optimal_params = bayes.bayes(
                badcase_bw,
                badcase_time,
                buffer_init=0,
                video_init=0,
                discount_factor=self.current_params['discount_factor']
            )
            
            self.data_manager.add_bw(badcase_bw,badcase_time,optimal_params)
            
            return {
                'status': 'success',
                'distribution_type': 'normal',
                'params': optimal_params,
                'message': 'First normal distribution pattern established'
            }
        
        # Analysis logic when normal distribution data is available
        current_stats = self._calculate_stats(badcase_bw)
        normal_stats = self._calculate_stats(normal_bw)
        is_normal, confidence = self._is_normal_distribution(badcase_bw, normal_stats)
        
        if is_normal and confidence > 0.8:
            bw_all = normal_bw + badcase_bw
            add_time = [badcase_time[i] - badcase_time[0] + normal_time[-1] for i in range(1, len(badcase_time))] 
            time_all = normal_time + add_time

            optimal_params = bayes.bayes(
                bw_all,
                time_all+ [time_all[-1]+1],
                buffer_init=0,
                video_init=0,
                discount_factor=self.current_params['discount_factor']
            )
            
            self.data_manager.add_bw(badcase_bw,badcase_time, optimal_params)
            
            return {
                'status': 'success',
                'distribution_type': 'normal',
                'params': optimal_params,
                'confidence': confidence
            }
    
        # Construct prompt phrases
        prompt = f"""Analyze the bandwidth distribution pattern and provide your analysis in the EXACT format below. Do not add any additional text.

                Normal Pattern: [yes/no]
                Matches Known Pattern: [yes/no]
                Best Matching Pattern: [pattern_id_number]
                Confidence: [0.0-1.0]
                Explanation: [your brief explanation]

                Current Pattern Statistics:
                - Sample size: {len(badcase_bw)}
                - Mean: {current_stats['mean']:.2f}
                - Standard deviation: {current_stats['std']:.2f}
                - CV: {current_stats['cv']:.2f}
                - Skewness: {current_stats['skew']:.2f}
                - Peak/Mean ratio: {current_stats['peak_mean_ratio']:.2f}
                """

        if normal_stats:
            prompt += f"""
            Normal Pattern Statistics:
            - Sample size: {len(normal_bw)}
            - Mean: {normal_stats['mean']:.2f}
            - Standard deviation: {normal_stats['std']:.2f}
            - CV: {normal_stats['cv']:.2f}
            - Skewness: {normal_stats['skew']:.2f}
            - Peak/Mean ratio: {normal_stats['peak_mean_ratio']:.2f}
            """

        if anomaly_patterns:
            prompt += "\nKnown Anomaly Patterns:\n"
            for pattern_id, pattern in anomaly_patterns.items():
                pattern_stats = self._calculate_stats(pattern['bw'])
                prompt += f"""
                Pattern {pattern_id}:
                - Mean: {pattern_stats['mean']:.2f}
                - Standard deviation: {pattern_stats['std']:.2f}
                - CV: {pattern_stats['cv']:.2f}
                - Skewness: {pattern_stats['skew']:.2f}
                - Peak/Mean ratio: {pattern_stats['peak_mean_ratio']:.2f}
                """

        prompt += """
            Please analyze the current bandwidth pattern and answer in the following format:
            1. Dose it close the normal distribution pattern? Answer "yes" or "no"
            2. If not, does it close any known anomaly pattern? Answer "yes" or "no"
            3. If it matches a known pattern, which pattern ID? Answer with the pattern ID
            4. What is your confidence level in this analysis? Answer with a number between 0 and 1

            Example answer format:
            Normal Pattern: no
            Matches Known Pattern: yes
            Best Matching Pattern: 2
            Confidence: 0.85
            Explanation: Brief explanation of the reasoning
            """

        # Obtain the analysis results from LLM
        llm_manager = get_llm_manager()
        response = llm_manager.get_raw_response(prompt)
        
        # Parse the response from llm
        analysis_result = self._parse_distribution_analysis(response)
        
        # Process the distribution based on the analysis results
        if analysis_result['is_normal']:
            # Normal distribution scenario
            bw_all = normal_bw + badcase_bw
            add_time = [badcase_time[i] - badcase_time[0] + normal_time[-1] for i in range(1, len(badcase_time))] 
            time_all = normal_time + add_time
            optimal_params = bayes.bayes(
                bw_all,
                time_all+ [time_all[-1]+1],
                buffer_init=0,
                video_init=0,
                discount_factor=self.current_params['discount_factor']
            )
            
            self.data_manager.add_bw(badcase_bw,badcase_time, optimal_params)
            self.data_manager.clear_pending_badcases()
            return {
                'status': 'success',
                'distribution_type': 'normal',
                'params': optimal_params
            }
        else:
            # # Abnormal distribution scenario
            if analysis_result['matches_known_pattern'] and analysis_result['pattern_id'] in anomaly_patterns:
                # Use parameters of a known pattern
                self.data_manager.clear_pending_badcases()
                return {
                    'status': 'success',
                    'distribution_type': 'anomaly',
                    'pattern_id': analysis_result['pattern_id'],
                    'params': anomaly_patterns[analysis_result['pattern_id']]['params'],
                    'confidence': analysis_result['confidence']
                }
            else:
                # Create a new abnormal pattern
                new_pattern_id = str(len(anomaly_patterns))
                optimal_params = bayes.bayes(
                    badcase_bw,
                    badcase_time,
                    buffer_init=0,
                    video_init=0,
                    discount_factor=self.current_params['discount_factor']
                )
                
                self.data_manager.add_anomaly_pattern(
                    new_pattern_id,
                    badcase_bw,
                    optimal_params
                )
                self.data_manager.clear_pending_badcases()
                return {
                    'status': 'success',
                    'distribution_type': 'anomaly',
                    'pattern_id': new_pattern_id,
                    'params': optimal_params,
                    'message': 'New anomaly pattern created'
                }

    def _parse_distribution_analysis(self, response: str) -> Dict:
        """Analyze the distribution analysis response from the LLM"""
        result = {
            'is_normal': False,
            'matches_known_pattern': False,
            'pattern_id': None,
            'confidence': 0.0,
            'explanation': ''
        }
        
        try:
            # Line-by-line processing
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            for line in lines:
                if ':' not in line:
                    continue
                    
                key, value = [part.strip().lower() for part in line.split(':', 1)]
                
                if key == 'normal pattern':
                    result['is_normal'] = value == 'yes'
                elif key == 'matches known pattern':
                    result['matches_known_pattern'] = value == 'yes'
                elif key == 'best matching pattern':
                    try:
                        # Extract numbers
                        pattern_id = ''.join(filter(str.isdigit, value))
                        if pattern_id:
                            result['pattern_id'] = pattern_id
                    except:
                        pass
                elif key == 'confidence':
                    try:
                        confidence = float(value)
                        result['confidence'] = max(0.0, min(1.0, confidence))
                    except:
                        pass
                elif key == 'explanation':
                    result['explanation'] = value
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            
        return result
    def _calculate_stats(self, data: List[float]) -> Dict:
        """Calculate statistical features"""
        if not data:
            return {}
            
        arr = np.array(data)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'skew': float(stats.skew(arr)),
            'kurtosis': float(stats.kurtosis(arr)),
            'peak_mean_ratio': float(np.max(arr) / np.mean(arr)),
            'cv': float(np.std(arr) / np.mean(arr)),
            'percentile_90': float(np.percentile(arr, 90))
        }
    def save_current_state(self):
        """Save the current state"""
        self.data_manager.save_patterns()

def generate_test_scenarios():
    """Generate data for various testing scenarios"""
    scenarios = []
    
    # Scenario 1: Bandwidth with normal fluctuations
    normal_cases = {
        'name': 'Normal Fluctuation',
        'data': [
            np.random.normal(100, 10, 20).tolist(),  # Stable network
            np.random.normal(80, 5, 20).tolist(),    # Lower bandwidth
            np.random.normal(120, 15, 20).tolist(),  # Higher bandwidth
        ]
    }
    scenarios.append(normal_cases)
    
    # Scenario 2: Sudden drop in bandwidth
    burst_drop_cases = {
        'name': 'Burst Drop',
        'data': [
            np.concatenate([
                np.random.normal(100, 5, 10),
                np.random.normal(30, 5, 10)
            ]).tolist(),
            np.concatenate([
                np.random.normal(90, 5, 10),
                np.random.normal(20, 5, 10)
            ]).tolist()
        ]
    }
    scenarios.append(burst_drop_cases)
    
    # Scenario 3: Periodic fluctuations
    periodic_cases = {
        'name': 'Periodic Fluctuation',
        'data': [
            (np.sin(np.linspace(0, 4*np.pi, 20)) * 20 + 100).tolist(),
            (np.sin(np.linspace(0, 4*np.pi, 20)) * 30 + 80).tolist()
        ]
    }
    scenarios.append(periodic_cases)
    
    # Scenario 4: Gradual drop
    gradual_drop_cases = {
        'name': 'Gradual Drop',
        'data': [
            np.linspace(100, 40, 20).tolist(),
            np.linspace(90, 30, 20).tolist()
        ]
    }
    scenarios.append(gradual_drop_cases)
    
    # Scenario 5: Highly unstable
    unstable_cases = {
        'name': 'Highly Unstable',
        'data': [
            np.random.exponential(50, 20).tolist(),
            np.random.gamma(2, 30, 20).tolist()
        ]
    }
    scenarios.append(unstable_cases)
    
    return scenarios

def offline_training_test():
    """Complete offline training and testing"""
    # Initialize the optimizer
    optimizer = ABROptimizer()
    
    # Obtain the testing scenario
    scenarios = generate_test_scenarios()
    
    # Record all discovered patterns
    discovered_patterns = set()
    all_results = []
    
    print("Starting Offline Training Test...")
    print("=" * 50)
    
    # Traverse each scenario
    for scenario in scenarios:
        print(f"\nProcessing Scenario: {scenario['name']}")
        print("-" * 30)
        
        # Process each data sequence within the scenario
        for i, bw_data in enumerate(scenario['data']):
            print(f"\nCase {i+1}:")
            print(f"Data length: {len(bw_data)}")
            print(f"Mean bandwidth: {np.mean(bw_data):.2f}")
            print(f"Std bandwidth: {np.std(bw_data):.2f}")
            
            # Process the data
            result = optimizer.process_badcase(bw_data)
            
            # Record the results
            result['scenario'] = scenario['name']
            result['case_id'] = i
            all_results.append(result)
            
            # Print a summary of the results
            print("\nAnalysis Result:")
            print(result)

    
    # Save the final pattern database
    optimizer.save_current_state()
    
    # Print the summary report
    print("\n" + "=" * 50)
    print("Training Summary:")
    print(f"Total scenarios processed: {len(scenarios)}")
    print(f"Total cases processed: {sum(len(s['data']) for s in scenarios)}")
    print(f"Unique patterns discovered: {len(discovered_patterns)}")
    print("\nPattern Distribution:")
    pattern_count = {}
    for result in all_results:
        if result['status'] == 'success':
            pattern_type = result.get('pattern_id', 'normal')
            pattern_count[pattern_type] = pattern_count.get(pattern_type, 0) + 1
    
    for pattern, count in pattern_count.items():
        print(f"Pattern {pattern}: {count} cases")
    
    # Validate the saved patterns
    print("\nVerifying saved patterns...")
    loaded_data = optimizer.data_manager.load_patterns()
    print(f"Number of saved patterns: {len(loaded_data['anomaly_patterns'])}")
    print("Saved pattern IDs:", list(loaded_data['anomaly_patterns'].keys()))

def main():
    """Main function"""
    print("Starting ABR Optimizer Offline Training...")
    print("=" * 50)
    
    # Execute the complete offline training and testing
    offline_training_test()
    
    print("\nOffline training completed.")
    print("=" * 50)

if __name__ == "__main__":
    main()