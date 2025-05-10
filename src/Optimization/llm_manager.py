import sys
sys.path.append("..") 
import llm_model.LLM as LLM
from typing import List, Dict, Any
import threading
class LLMManager:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if LLMManager._instance is not None:
            raise Exception("This class is a singleton. Use get_instance() instead.")
        self.predictor = LLM.BandwidthPredictor()
    
    def predict_bandwidth(self, bw: List[float], 
                         duration_series: List[float], 
                         timestamp_series: List[float]) -> Any:
        """Predict bandwidth"""
        bw = [i/8 for i in bw]
        outputs =  self.predictor.predict(bw, duration_series, timestamp_series)
        outputs = [i *8 for i in outputs]
        return outputs
    
    def evaluate_algorithms(self, prompt: str, algorithm_keys: List[str]) -> tuple:
        """evaluate algorithm"""
        return self.predictor.LLMout(prompt, algorithm_keys)
    
    def get_raw_response(self, prompt: str) -> str:
        """Gett the raw LLM response"""
        return self.predictor.RAWLLMout(prompt)

# global access function
def get_llm_manager():
    return LLMManager.get_instance()