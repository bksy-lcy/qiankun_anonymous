import numpy as np
import fixed_env_mpc as env
import os
import itertools
import sys
import json
from typing import List, Dict, Tuple
from scipy import stats
from data import DataManager
import load_trace
def str_to_list(s):
    s = s.strip('[]')
    return [float(x) for x in s.split(',')]
class RuleBayesianDetector:
    def __init__(self, features, thresholds, weights, distributions):
        self.features = features
        self.thresholds = thresholds
        self.weights = weights
        self.distributions = distributions
        self.priors = self._initialize_priors()
        
    def _initialize_priors(self) -> Dict:
        """Initialize prior distribution parameters"""
        priors = {}
        for feature, dist_type in self.distributions.items():
            if dist_type == 'normal':
                priors[feature] = {
                    'mu_0': (self.thresholds[feature]['low'] + 
                            self.thresholds[feature]['high']) / 2,
                    'kappa_0': 1,
                    'alpha_0': 1,
                    'beta_0': 1
                }
            elif dist_type in ['gamma', 'exponential']:
                priors[feature] = {
                    'alpha': 2,
                    'beta': 1
                }
        return priors
        
    def detect(self, features: Dict[str, float]) -> bool:
        """Test for normal distribution"""
        probabilities = []
        
        for feature, value in features.items():
            if feature not in self.features:
                continue
                
            prob = self._compute_posterior_probability(
                feature, 
                value, 
                self.distributions[feature],
                self.priors[feature]
            )
            
            weighted_prob = prob * self.weights[feature]
            probabilities.append(weighted_prob)
            
            self._update_prior(feature, value)
            
        final_prob = np.mean(probabilities)
        return final_prob > 0.5
        
    def _compute_posterior_probability(self, 
                                    feature: str, 
                                    value: float,
                                    dist_type: str,
                                    prior: Dict) -> float:
        """Calculate posterior probability"""
        if dist_type == 'normal':
            mu = prior['mu_0']
            beta = prior['beta_0']
            alpha = prior['alpha_0']
            kappa = prior['kappa_0']
            
            nu = 2 * alpha
            sigma = np.sqrt(beta * (1 + 1/kappa) / alpha)
            prob = 1 - stats.t.cdf(abs(value - mu) / sigma, nu)
            
        elif dist_type in ['gamma', 'exponential']:
            alpha = prior['alpha']
            beta = prior['beta']
            prob = stats.gamma.pdf(value, alpha, scale=1/beta)
            
        return prob
        
    def _update_prior(self, feature: str, value: float):
        """Update prior distribution parameters"""
        dist_type = self.distributions[feature]
        prior = self.priors[feature]
        
        if dist_type == 'normal':
            kappa_n = prior['kappa_0'] + 1
            mu_n = (prior['kappa_0'] * prior['mu_0'] + value) / kappa_n
            alpha_n = prior['alpha_0'] + 0.5
            beta_n = (prior['beta_0'] + 
                     0.5 * prior['kappa_0'] * 
                     (value - prior['mu_0'])**2 / kappa_n)
            
            self.priors[feature].update({
                'kappa_0': kappa_n,
                'mu_0': mu_n,
                'alpha_0': alpha_n,
                'beta_0': beta_n
            })
            
        elif dist_type in ['gamma', 'exponential']:
            alpha_n = prior['alpha'] + 1
            beta_n = prior['beta'] + value
            
            self.priors[feature].update({
                'alpha': alpha_n,
                'beta': beta_n
            })
class BayesianDistributionDetector:
    def __init__(self, dataset,file_name):
        self.data_manager = DataManager( dataset,file_name)
        self.data_manager.load_patterns()
        self.rules = self._initialize_rules()
        self.bayes_detector = self._initialize_bayes_detector(self.rules)
    def _initialize_bayes_detector(self, rules: Dict) -> 'RuleBayesianDetector':
        """Initialize Bayesian detector"""
        return RuleBayesianDetector(
            features=rules['features'],
            thresholds=rules['thresholds'],
            weights=rules['weights'],
            distributions=rules['distributions']
        )
        
    def _initialize_rules(self) -> Dict:
        """Initialize detection rules"""
        rules = {
            'features': ['mean', 'std', 'cv', 'peak_mean_ratio'],
            'thresholds': {},
            'weights': {},
            'distributions': {}
        }
        
        # Calculate thresholds from normal data
        if self.data_manager.data['normal_bw']:
            normal_features = self._calculate_features(self.data_manager.data['normal_bw'])
            
            # Set thresholds
            for feature in rules['features']:
                value = normal_features[feature]
                rules['thresholds'][feature] = {
                    'low': value * 0.8,
                    'high': value * 1.2
                }
        else:
            # Use default thresholds
            rules['thresholds'] = {
                'mean': {'low': 0.8, 'high': 1.2},
                'std': {'low': 0.1, 'high': 0.3},
                'cv': {'low': 0.1, 'high': 0.5},
                'peak_mean_ratio': {'low': 1.0, 'high': 2.0}
            }
        
        # Set weights and distribution types
        rules['weights'] = {
            'mean': 0.3,
            'std': 0.3,
            'cv': 0.2,
            'peak_mean_ratio': 0.2
        }
        
        rules['distributions'] = {
            'mean': 'normal',
            'std': 'gamma',
            'cv': 'gamma',
            'peak_mean_ratio': 'gamma'
        }
        
        return rules
    def _calculate_features(self, data: List[float]) -> Dict[str, float]:
        """Calculate the features of the bandwidth sequence"""
        mean = np.mean(data)
        std = np.std(data)
        return {
            'mean': mean,
            'std': std,
            'cv': std / mean if mean > 0 else 0,
            'peak_mean_ratio': np.max(data) / mean if mean > 0 else 1
        }

    def check_distribution(self, bw_sequence: List[float]) -> Tuple[bool, int]:
        """Examine the distribution type of the bandwidth sequence"""
        features = self._calculate_features(bw_sequence)
        is_normal = self.bayes_detector.detect(features)
        
        if not is_normal:
            # Check for matches with known anomaly patterns
            best_pattern = self._match_anomaly_pattern(features)
            return False, best_pattern
        
        return True, None
        
    def _match_anomaly_pattern(self, features: Dict[str, float]) -> int:
        """Match the optimal anomaly pattern"""
        best_match = None
        best_similarity = float('-inf')
        
        for pattern_id, pattern_data in self.data_manager.data['anomaly_patterns'].items():
            pattern_features = self._calculate_features(pattern_data['bw'])
            similarity = self._calculate_similarity(features, pattern_features)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = int(pattern_id)
                
        return best_match
        
    def _calculate_similarity(self, features1: Dict[str, float], 
                            features2: Dict[str, float]) -> float:
        """Calculate feature similarity"""
        similarity = 0
        for feature in self.rules['features']:
            if feature in features1 and feature in features2:
                weight = self.rules['weights'][feature]
                diff = abs(features1[feature] - features2[feature])
                max_val = max(features1[feature], features2[feature])
                if max_val > 0:
                    similarity += weight * (1 - diff/max_val)
        return similarity

class AdaptiveDiscountOptimizer:
    def __init__(self, dataset,file_name, initial_discount_factor=1.0):
        self.distribution_detector = BayesianDistributionDetector( dataset,file_name)
        self.data_manager = self.distribution_detector.data_manager
        self.params = self._load_params(initial_discount_factor)
        self.bandwidth_window = []
        self.window_size = 20
    def _load_params(self, initial_discount_factor: float) -> Dict[str, float]:
        """Load discount factor parameters"""
      

        params = {
            'normal': initial_discount_factor,
            'badcase': initial_discount_factor
        }
        return params
    def update_bandwidth(self, bandwidth: float):
        """Update the bandwidth window
        
        Args:
            bandwidth: New bandwidth measurement value
        """
        self.bandwidth_window.append(bandwidth)
        if len(self.bandwidth_window) > self.window_size:
            self.bandwidth_window.pop(0)
    def detect_distribution(self) -> str:
        """Detect the current bandwidth distribution type"""
        if len(self.bandwidth_window) < self.window_size:
            return 'normal'
            
        is_normal, pattern_id = self.distribution_detector.check_distribution(self.bandwidth_window)
        
        if is_normal:
            return 'normal'
        elif pattern_id is not None:
            return f'pattern_{pattern_id}'
        else:
            return 'badcase'
            
    def get_discount_factor(self) -> float:
        """Obtain the current discount factor"""
        dist_type = self.detect_distribution()
        if dist_type.startswith('pattern_'):
            pattern_id = int(dist_type.split('_')[1])
            return self.data_manager.data['anomaly_patterns'][str(pattern_id)]['params']

        if dist_type == 'normal':
            if self.data_manager.data['normal_params'] is None:
                return 1
            else:
                return  self.data_manager.data['normal_params'] 
        else:
            return 1
size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]

def get_chunk_size(quality, index):
    if (index < 0 or index > 48):
        return 0
    sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index],
             2: size_video4[index], 1: size_video5[index], 0: size_video6[index]}
    return sizes[quality]


buffer_init = 0
video_init = 0
test_trace = sys.argv[1]
SUMMARY_DIR = './results'
LOG_FILE = './results/' +test_trace+ '/log_sim_b2ayes'
COOKED_TRACE_FOLDER = '/test/'+test_trace
if not os.path.exists('./results/' +test_trace):
    os.mkdir('./results/' +test_trace)
def main():
    current_discount_factor = 1
    
    # Constants
    S_INFO = 5
    S_LEN = 8
    A_DIM = 6
    RANDOM_SEED = 42
    VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
    M_IN_K = 1000.0
    REBUF_PENALTY = 4.3
    SMOOTH_PENALTY = 1
    DEFAULT_QUALITY = 1
    CHUNK_COMBO_OPTIONS = []
    TOTAL_VIDEO_CHUNKS = 48
    CHUNK_TIL_VIDEO_END_CAP = 48.0
    BUFFER_NORM_FACTOR = 10.0
    MPC_FUTURE_CHUNK_COUNT = 5

    # Initialize
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM
    
    # Initialize environment and optimizer
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(COOKED_TRACE_FOLDER)
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                            all_cooked_bw=all_cooked_bw,
                            buffer_init=buffer_init,
                            video_init= video_init)
       



    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')
    adaptive_optimizer = AdaptiveDiscountOptimizer(
         test_trace,all_file_names[net_env.trace_idx],
        initial_discount_factor=current_discount_factor
    )


    # Initialize other variables
    time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1
    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    video_count = 0
    
    # Initialize QoE metrics
    qoe_metrics = {
        'rewards': [],
        'rebuffers': [],
        'smoothness': []
    }

    # Generate all possible chunk combinations
    for combo in itertools.product([0,1,2,3,4,5], repeat=5):
        CHUNK_COMBO_OPTIONS.append(combo)

    # past errors in bandwidth
    past_errors = []
    past_bandwidth_ests = []

    while True:  # serve video forever
        # Get next video chunk
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain, end_of_bw = \
            net_env.get_video_chunk(bit_rate)

        # Update time
        time_stamp += delay
        time_stamp += sleep_time

        # Calculate reward
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                        VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        r_batch.append(reward)
        last_bit_rate = bit_rate
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                    str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                    str(buffer_size) + '\t' +
                    str(rebuf) + '\t' +
                    str(video_chunk_size) + '\t' +
                    str(delay) + '\t' +
                       str(current_discount_factor) + '\t' +
                    str(reward) + '\n')
        log_file.flush()
        # Update state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        state = np.roll(state, -1, axis=1)

        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K
        state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP

        # Update bandwidth measurement
        bandwidth = float(video_chunk_size) / float(delay) / M_IN_K
        adaptive_optimizer.update_bandwidth(bandwidth)

        # Compute current error and update past errors
        curr_error = 0
        if len(past_bandwidth_ests) > 0:
            curr_error = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
        past_errors.append(curr_error)

        # Calculate harmonic mean of past bandwidths
        past_bandwidths = state[3,-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        # Get current discount factor
        current_discount_factor = adaptive_optimizer.get_discount_factor()
        # Estimate future bandwidth
        future_bandwidth = harmonic_bandwidth/(1+current_discount_factor)
        past_bandwidth_ests.append(harmonic_bandwidth)

        # MPC algorithm
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if (TOTAL_VIDEO_CHUNKS - last_index < 5):
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

        max_reward = float('-inf')
        best_combo = ()
        start_buffer = buffer_size

        for full_combo in CHUNK_COMBO_OPTIONS:
            combo = full_combo[0:future_chunk_length]
            curr_rebuffer_time = 0
            curr_buffer = start_buffer
            bitrate_sum = 0
            smoothness_diffs = 0
            last_quality = int(bit_rate)

            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = last_index + position + 1
                download_time = (get_chunk_size(chunk_quality, index)/1000000.)/future_bandwidth

                if curr_buffer < download_time:
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4

                bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                last_quality = chunk_quality

            reward = (bitrate_sum/1000.) - (REBUF_PENALTY*curr_rebuffer_time) - (smoothness_diffs/1000.)

            if reward >= max_reward:
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward

        bit_rate = best_combo[0] if best_combo else 0

      

        s_batch.append(state)

        if end_of_video:
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            video_count += 1
        if end_of_bw:
            log_file.write('\n')
            log_file.close()
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY

           

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []
            current_discount_factor = 1
            
            if net_env.trace_idx >= len(all_file_names):
                print(test_trace,len(all_file_names))
                break
            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')
            print(net_env.trace_idx,all_file_names[net_env.trace_idx])
            adaptive_optimizer = AdaptiveDiscountOptimizer(
            test_trace,all_file_names[net_env.trace_idx],
            initial_discount_factor=current_discount_factor
        )

if __name__ == '__main__':
    main()