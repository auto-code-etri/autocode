import sys
import os
from pprint import pprint

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath("third_party/etri_langgraph"))

from networks.archcode.archcode import ArchCodeNet
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    net = ArchCodeNet()
    net.compile()
    result = net.run(state={'input_data': "implement sorting algorithm", "code_n": 10})
    
    print("---------------------------------------------------")
    pprint(result)
    print("---------------------------------------------------")
    
    top_k = 1
    top_k_results = result["formatted_result"][:top_k]
    
    print("Top {} results: ".format(top_k))
    pprint(top_k_results)
    print()