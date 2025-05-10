from typing import List, Dict, Tuple
import os
import json
import numpy as np
class DataManager:
    def __init__(self, dataset,file_name):
        self.storage_path = 'bayes_rules/'+dataset
        self.file_name = file_name
        self.data = {
            'normal_bw': [],
             'normal_time': [],
            'normal_params':1,
            'anomaly_patterns': {},  # {pattern_id: {'bw': [], 'params': {}}}
            'pending_badcases': []
        }
        self.start_time = 0
    def load_patterns(self) -> bool:
        """
        Load pattern database from file
        
        Args:
            filename: Database filename
            
        Returns:
            bool: whether loading was successful or not
        """
        filename = self.file_name
        full_path = os.path.join(self.storage_path, filename)
        
        try:
            if not os.path.exists(full_path):
                print(f"Pattern database file not found: {full_path}")
                return False
                
            with open(full_path, 'r') as f:
                loaded_data = json.load(f)
            # Validate the format of the loaded data
            if not self._validate_loaded_data(loaded_data):
                print("Invalid data format in pattern database")
                return False
                
            # Updated data
            self.data.update(loaded_data)
            return self.data
            
        except Exception as e:
            print(f"Error loading pattern database: {e}")
            return False
            
    def _validate_loaded_data(self, data: Dict) -> bool:
        """
        Validate the format of the loaded data
        
        Args:
            data: Data to be validated
            
        Returns:
            bool: Validity of the data format
        """
        required_keys = {'normal_bw', 'anomaly_patterns', 'pending_badcases'}
        
        # Check the required keys
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            print(f"Missing required keys: {missing_keys}")
            return False
            
        # Validate Data Type
        if not isinstance(data['normal_bw'], list):
            print("normal_bw is not a list")
            return False
            
        if not isinstance(data['anomaly_patterns'], dict):
            print("anomaly_patterns is not a dict")
            return False
        
        # Validating Anomaly Pattern Format
        for pattern_id, pattern_data in data['anomaly_patterns'].items():
            if not isinstance(pattern_data, dict):
                print(f"Data of pattern {pattern_id} is not a dict")
                return False
                
            if 'bw' not in pattern_data:
                print(f"pattern {pattern_id} Missing 'bw' field")
                return False
                
            if 'params' not in pattern_data:
                print(f"pattern {pattern_id} Missing 'params' field")
                return False
                
            if not isinstance(pattern_data['bw'], list):
                print(f"'bw' of pattern {pattern_id} is not a list")
                return False
                
            if not isinstance(pattern_data['params'], float):
                print(f"'params' of pattern {pattern_id} is not a dict")
                return False
                
        if not isinstance(data['pending_badcases'], list):
            print("pending_badcases is not a list")
            return False
            
        return True
        
    def add_bw(self, bw,time,params = None):
        """Add normal bandwidth data"""
        self.data['normal_bw'].extend(bw)
     
        new_time = [time[i]-time[0]+self.start_time for i in range(1,len(time))]
        new_time = new_time + [new_time[-1]+1]
        self.data['normal_time'] += new_time 
        self.start_time = self.data['normal_time'][-1]
        if params is not None:
            self.data['normal_params'] = params
        
    def add_anomaly_pattern(self, pattern_id: int, bw_data: List[float], params: Dict):
        """Add new anomaly pattern"""
        self.data['anomaly_patterns'][str(pattern_id)] = {
            'bw': bw_data,
            'params': params
        }
        
    def get_all_data(self) -> Dict:
        """get all data"""
        return self.data
        
    def clear_pending_badcases(self):
        """clear pending badcases"""
        self.data['pending_badcases'] = []
    def save_patterns(self):
        """save pattern database"""
        full_path = os.path.join(self.storage_path, self.file_name)
        
        # Converting numpy arrays to lists for JSON serialization
        serializable_data = self._make_serializable(self.data)
        
        with open(full_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
            
        
    def _make_serializable(self, data: Dict) -> Dict:
        """Converting data to a serializable format"""
        serializable = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                serializable[key] = self._make_serializable(value)
            elif isinstance(value, list):
                serializable[key] = [float(x) if isinstance(x, np.number) else x for x in value]
            elif isinstance(value, np.number):
                serializable[key] = float(value)
            else:
                serializable[key] = value
                
        return serializable
def main():
    data_manager = DataManager()
    loaded_data = data_manager.load_patterns()
    print(loaded_data)
    print(f"Number of saved patterns: {len(loaded_data['anomaly_patterns'])}")
    print("Saved pattern IDs:", list(loaded_data['anomaly_patterns'].keys()))
if __name__ == "__main__":
    main()