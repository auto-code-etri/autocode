from networks.archcode.utils.parser import CodeBlockParser, DistributeParser, FindParser
from src.utils.registry import node_registry, BaseNode
from src.prompt import *
from src.model import *

from langchain_core.output_parsers import StrOutputParser
import concurrent.futures
import json
import os
import requests



@node_registry(name="executionresultformatter")
class ExecutionResultFormatter(BaseNode):
    """
    Format execution result.
    inputs:
        - exec_result: dict
    outputs:
        - formatted_result: dict
    """
    def __init__(self, **data):
        super().__init__(**data)      
    
    def __call__(self, state: dict) -> dict:
        print(self.get_name())
        
        codes = state.get('codes', [])
        exec_results = state.get('exec_result', {})
        
        formatted_result = []
        for idx, code in enumerate(codes):
            exec_result = exec_results.get(idx, {})
            # exec_result overall_score
            pass_counts = exec_result.get("pass_counts", {})
            total_counts = exec_result.get("total_counts", {})
            
            frs = ["general", "edge"]
            fr_score = {fr: pass_counts.get(fr, 0) / total_counts.get(fr, 0) if total_counts.get(fr, 0) > 0 else 0 for fr in frs}
            
            nfrs = ["performance", "sqr", "robustness", "maintainability"]
            nfr_score = {nfr: pass_counts.get(nfr, 0) / total_counts.get(nfr, 0) if total_counts.get(nfr, 0) > 0 else 0 for nfr in nfrs}
            
            overall_score = sum(pass_counts.values()) / sum(total_counts.values()) if len(total_counts.values()) > 0 else 0
            formatted_result.append({
                'code': code,
                # 'exec_result': exec_result,
                'overall_score': overall_score,
                'fr': fr_score,
                'nfr': nfr_score
            })

        # sort by overall_score
        formatted_result.sort(key=lambda x: x['overall_score'], reverse=True)

        state['formatted_result'] = formatted_result
        return state