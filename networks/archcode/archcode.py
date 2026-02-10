from src.network.network import Network
from networks.archcode.nodes.generators import *
from networks.archcode.nodes.executors import *
from networks.archcode.nodes.formatter import *

from etri_langgraph.config import Config

from langgraph.graph import StateGraph
from langchain_core.messages import AnyMessage

from operator import add
from typing import Optional, Any, TypedDict, Annotated, Iterable

from pprint import pprint
from deepmerge import always_merger

import os





# for parallelization utils
take_last = lambda a, b: b
def merge(a: Any, b: Any) -> Iterable[Any]:
    a_type = type(a)
    b_type = type(b)
    
    if a is None:
        return b
    if b is None:
        return a

    assert a_type == b_type, "Type mismatch: {} != {}".format(a_type, b_type)

    if isinstance(a, list) and isinstance(b, list):
        return a + b
    elif isinstance(a, dict) and isinstance(b, dict):
        return always_merger.merge(a, b)
    else:
        return [a, b]


class ArchCodeState(TypedDict):
    input_data: Annotated[Any, take_last]
    code_n: Annotated[int, take_last]
    input_prompt: Annotated[Any, take_last]
    
    requirements: Annotated[Any, take_last]
    plan: Annotated[Any, take_last]
    
    # parallelization need merging
    codes: Annotated[Any, merge]
    func_testcase: Annotated[Any, merge]
    nonfunc_testcase: Annotated[Any, merge]
    
    exec_result: Annotated[dict, take_last]
    formatted_result: Annotated[dict, take_last]

class ArchCodeNet(Network):
    network_name: str = "ArchCode"
    relative_path: str = "networks/archcode"
    config: Optional[Any] = None
    
    def __init__(self, **data):
        super().__init__(**data)

    def compile(self):
        yaml_graph_path = f"{self.relative_path}/{self.network_name}_etri.yaml"
        graph_config, node_functions = self.gather_graph_info(yaml_graph_path)
        
        # Use StateGraph with our custom TypedDict
        builder = StateGraph(ArchCodeState)
        
        nodes = graph_config.graph.nodes
        for node in nodes:
            func = node_functions[node.name]
            builder.add_node(node.name, func)
            
        builder.set_entry_point(graph_config.graph.entry_point)

        edges = graph_config.graph.edges
        for edge in edges:
            pair = edge.pair
            if edge.type == "always":
                builder.add_edge(pair[0], pair[1])
        
        self.graph = builder.compile()
        # self.graph.get_graph().draw_mermaid_png(output_file_path="./archcode.png")

        return self.graph