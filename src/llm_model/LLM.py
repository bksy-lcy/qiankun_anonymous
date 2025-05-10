import numpy as np
import torch
from llm_model.config import get_args
from llm_model.models import model
import joblib
import time 
import json
import resource
import psutil
from copy import deepcopy
from tqdm import tqdm
import gc

class BandwidthPredictor:
    def __init__(self):
        self.model = None
        self.args = get_args()
        self.model_file = 'your_path'
        self.scaler = None
        self.input_len = 4
        self.output_len = 5
            
        # Set the random seed
        fix_seed = 2021
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        self.mark = torch.load(self.model_file + '/16.pt')
        self.init_model()
    
    def init_model(self):
        
        self.model = model.Model(self.args)
        checkpoint = torch.load(self.model_file + "/checkpoint.pth")
        self.model.load_state_dict(checkpoint, strict=False)
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def predict(self, past_bandwidths, bw_timestamp, bw_duration):
        # Standardize the input data format
        self.token_len = 5
        self.output_len = len(past_bandwidths)
        token_number = int(np.ceil(self.output_len / self.token_len) ) 
        self.input_len = len(past_bandwidths)
        
        def ensure_2d_array(data):
            if isinstance(data, list):
                data = np.array(data)
            return data.reshape(1, -1) if len(data.shape) == 1 else data

        past_bandwidths = ensure_2d_array(past_bandwidths)
        bw_timestamp = ensure_2d_array(bw_timestamp)  
        bw_duration = ensure_2d_array(bw_duration)
        self.input_len = int(np.ceil(past_bandwidths.shape[-1] /self.token_len))*self.token_len
        input_token_number = int(np.ceil(past_bandwidths.shape[-1] /self.token_len))
        def pad_sequence(values, pad_len=5):

            batch_size = values.shape[0]
     
            padded_values = np.zeros((batch_size, pad_len))
            for i in range(batch_size):
                # Obtain the sequence of the current batch
                current_sequence = values[i].flatten()  # Flatten the sequence of the current batch
                sequence = list(current_sequence)
                
                # Obtain non-zero values
                non_zeros = [x for x in sequence if x != 0]
                last_non_zero = non_zeros[-1] if non_zeros else 0
                
                # Fill to pad_len using the last non-zero value
                while len(sequence) < pad_len:
                    sequence.append(last_non_zero)
                padded_values[i] = sequence
            
            return padded_values  
            

        with torch.no_grad():
            padded_bandwidths = pad_sequence(past_bandwidths,self.input_len )
            batch_x = np.array(padded_bandwidths).reshape(-1, self.input_len, 1)
            batch_x = torch.tensor(batch_x).to(dtype=torch.float32).to(self.device).float()
            batch_x = torch.log1p(batch_x.float())
            extended_timestamp = pad_sequence(bw_timestamp, self.input_len)
            extended_duration = pad_sequence(bw_duration, self.input_len)
         
            batch_x_mark =  np.stack((extended_timestamp, extended_duration), axis=2)
            def transform_batch_data(batch_x_mark):
                transformed_data = torch.zeros_like(batch_x_mark)
                
                # Transformation of the second dimension: Directly compute Log(x)/10
                transformed_data[:, :, 1] = torch.log1p(batch_x_mark[:, :, 1]) / 10
                
                # Transformation of the first dimension: Compute the difference relative to the minimum value, followed by Log(x)/10
                dim2_data = batch_x_mark[:, :, 0]
                for batch_idx in range(batch_x_mark.shape[0]):
                    current_data = dim2_data[batch_idx]
                    
                    # Calculate the difference relative to the minimum value
                    min_val = torch.min(current_data)
                    diff = current_data - min_val
                    
                    # Perform a logarithmic transformation on the differences
                    transformed_data[batch_idx, :, 0] = torch.log1p(diff/1000) / 10
                
                return transformed_data

            batch_x_mark = torch.tensor(batch_x_mark).to(self.device)
            batch_x_mark = transform_batch_data(batch_x_mark).float()
          
            
            device = next(self.model.parameters()).device
         
            outputs = []
            current_input =torch.cat((batch_x, batch_x_mark[:,:self.input_len,:]), dim=2)
            self.this_mark = self.mark[:input_token_number].reshape(1, input_token_number, -1).repeat(current_input.shape[0], 1, 1)

            for current_idx in range(token_number):
                start_time = time.time()
                with torch.cuda.amp.autocast():
                    output = self.model(deepcopy(current_input),self.this_mark.to(self.device), '', '')
                    end_time = time.time()
                    current_input = torch.cat([current_input[:, self.token_len:,0:1], output[:,:self.token_len, 0:1]], dim=1)
                    current_input =torch.cat((current_input, batch_x_mark[:,:self.input_len,:]), dim=2).to(self.device).float()
                    outputs.append(output[:, :self.token_len,0:1].unsqueeze(1).cpu().numpy())
            outputs =  (np.exp(outputs) - 1)*1000
            outputs = np.round(outputs).astype(int).flatten().tolist()
            gc.collect()
            torch.cuda.empty_cache()
            
            return outputs
    def parse_llm_output(self, output_text, algs=None) -> tuple:
        try:
            # Default
            algorithm = None
            confidence = None
            need_more_data = None  # Variables added for the second scenario
            
            # Clean Output Text
            # 1. Remove excessive whitespace characters
            output_text = ' '.join(output_text.split())
            # 2. Replace potential variants
            output_text = output_text.replace('：', ':').replace('，', ',')
            
            # Use regular expressions to find algorithms and confidence levels
            import re
            
            # find the algorithm name (matching the first algorithm name after "Best algorithm")
            # If the algs parameter is not provided, use the default value
            if algs is None:
                algs = ['bb', 'mpc', 'rl_no_training']
                return_multiple = False  # Return only one algorithm
            else:
                return_multiple = True   # Return multiple algorithms sorted in order
                
            for line in output_text.split('\n'):
                if 'Best algorithm' in line:
                    # Record the position of each algorithm in the line
                    positions = {}
                    for alg in algs:
                        pos = line.find(alg)
                        if pos != -1:  # If an algorithm is found
                            positions[alg] = pos
                    
                    # If an algorithm is found
                    if positions:
                        # Sort the algorithms based on their position in the text
                        sorted_algorithms = sorted(positions.items(), key=lambda x: x[1])
                        if return_multiple:
                            # Extract all algorithm names in sorted order
                            algorithm = [alg[0] for alg in sorted_algorithms]
                        else:
                            # Select only the algorithm that is positioned the earliest
                            algorithm = sorted_algorithms[0][0]
                    break
                    
            # Locate the confidence level (the first floating-point number following the term "Confidence level").
            conf_match = re.search(r'Confidence level\s*:?\s*([-+]?\d*\.?\d+)', output_text)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                    # Ensure that the confidence level is between 0 and 1
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = None
            
            # Only check for the need for more data in the case of the second type of invocation
            if return_multiple:
                # Check for the need for more data (matching the response following the phrase "Whether need more data").
                data_match = re.search(r'Whether need more data\s*:?\s*([YyNn]|[Yy]es|[Nn]o)', output_text, re.IGNORECASE)
                if data_match:
                    answer = data_match.group(1).lower()
                    if answer in ['y', 'yes']:
                        need_more_data = 'Y'
                    elif answer in ['n', 'no']:
                        need_more_data = 'N'
            
            # Return different results based on the type of invocation
            if return_multiple:
                return algorithm, confidence, need_more_data
            else:
                return algorithm, confidence
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print(f"Raw output: {output_text}")
            # Return distinct results according to the context of the invocation
            if algs is None:  # First scenario
                return None, None
            else:  # Second scenario
                return None, None, None

    def LLMout(self, prompt, algs=None):
        if algs is None:
            # First scenario
            output = self.model.LLMout(prompt)
            print('LLMout', output)
            algorithm, confidence = self.parse_llm_output(output)
            return algorithm, confidence
        else:
            # Second scenario
            algorithm = None
            while algorithm == None:
                output = self.model.LLMout([prompt])
                algorithm, confidence, need_more_data = self.parse_llm_output(output, algs)
            return algorithm, confidence, need_more_data

    def RAWLLMout(self,prompt):
        output = self.model.LLMout([prompt])
        return output
    def parse_parameters(self,response_text):
        """
        Parse parameters from the response text
        
        Args:
            response_text: String containing the response with format:
                Parameters: X.XX
                Confidence: Y.YY
                
        Returns:
            float: The parameter value, or None if parsing fails
        """
        try:
            # Split text into lines and find Parameters line
            lines = response_text.strip().split('\n')
            for line in lines:
                if line.startswith('Parameters:'):
                    # Extract the first number before the comma
                    param_str = line.replace('Parameters:', '').strip()
                    try:
                        # Split by comma and take first value
                        param_value = float(param_str.split(',')[0].strip())
                        # Validate range
                        if 0 <= param_value <= 1:
                            return param_value
                        else:
                            print(f"Parameter {param_value} outside valid range 0-1")
                            return None
                    except ValueError as e:
                        print(f"Error converting to float: {e}")
                        return None
            
            print("No valid Parameters line found")
            return None
            
        except ValueError:
            print("Failed to convert parameter to float")
            return None
        except Exception as e:
            print(f"Error parsing parameters: {str(e)}")
            return None

    def LLMimprove(self,prompt):
        parameter = None
        while parameter== None:
            output = self.model.LLMout(prompt)
            parameter = self.parse_parameters(output)
            print('improve:',output,parameter)
        return parameter