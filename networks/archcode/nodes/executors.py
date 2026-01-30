from networks.archcode.utils.parser import CodeBlockParser, DistributeParser, FindParser
from src.utils.registry import node_registry, BaseNode
from src.prompt import *
from src.model import *

from langchain_core.output_parsers import StrOutputParser
import concurrent.futures
import json
import os
import requests



@node_registry(name="pythoncodeexecutor")
class PythonCodeExecutor(BaseNode):
    """
    Execute Python code.
    inputs:
        - code: str
        - tc: str
    outputs:
        - exec_result: str
    """
    def __init__(self, **data):
        super().__init__(**data)      
    
    def _execute_single_case(self, code_snippet, assert_code):
        full_code = code_snippet + "\n\n" + assert_code
        try:
            response = requests.post(
                os.environ["CODEEXEC_ENDPOINT"],
                data=json.dumps({
                    "code": full_code,
                    "timeout": 3
                }),
                headers={"Content-Type": "application/json"},
            )
            execution_result = response.json().get("output", "")
            return "Exit Code: 0" in execution_result
        except Exception:
            return False

    def __call__(self, state: dict) -> dict:
        print(self.get_name())
        
        codes = state.get('codes', [])
        testcases = {}
        
        if 'func_testcase' in state and state['func_testcase']:
            testcases.update(state['func_testcase'])
        if 'nonfunc_testcase' in state and state['nonfunc_testcase']:
            testcases.update(state['nonfunc_testcase'])
        
        if not os.environ.get("CODEEXEC_ENDPOINT"):
            print("WARNING: CODEEXEC_ENDPOINT not set. Skipping execution.")
            state['exec_result'] = ["Skipped Execution"]
            return state
        
        assert_parser = FindParser(
            patterns=[
                # Capture block starting with "### Testcase <number>:" until next test case or end of string
                r"### Testcase \d+:[\s\S]*?(?=\n### Testcase \d+:|\Z)",
            ]
        )
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for idx, code in enumerate(codes):
                target_results_rate = {}
                target_results_pass_counts = {}
                target_results_total_counts = {}
                
                # We will collect futures first to keep structure clean
                # But to wait for all, we might want to just process one code block's tests at a time 
                # or verify if we want to dump ALL tasks for ALL codes into the pool.
                # Dumping all tasks is better for utilization.
                
                type_futures = {} # mapped by tc_type -> list of futures
                
                for tc_type, tcs in testcases.items():
                    assert_codes = assert_parser.invoke(tcs)
                    # Submit all tasks for this type
                    type_futures[tc_type] = [executor.submit(self._execute_single_case, code, ac) for ac in assert_codes]
                    target_results_total_counts[tc_type] = len(assert_codes)
                
                # Now collect results for this code index
                for tc_type, futures in type_futures.items():
                    # Wait for all futures of this type to complete
                    pass_count = sum(1 for f in concurrent.futures.as_completed(futures) if f.result())
                    
                    target_results_pass_counts[tc_type] = pass_count
                    total = target_results_total_counts[tc_type]
                    target_results_rate[tc_type] = round(pass_count / total * 100, 2) if total > 0 else None
                
                results[idx] = {
                    'pass_counts': target_results_pass_counts,
                    'total_counts': target_results_total_counts,
                    'rate': target_results_rate
                }

        state['exec_result'] = results
        return state