from typing import Dict, List, Callable
from src.network.config import GraphConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import Callable
from typing import List

class Graph(BaseModel):
    config: GraphConfig

    def __init__(self, **data):
        super().__init__(**data)
    
    def compose_and_compile(self, node_functions: Dict[str, Callable]):
        builder = StateGraph(List[dict])    #langgraph
        nodes = self.config.nodes    #node 설정
        for node in nodes:      # initialize, excute node function
            func = node_functions[node.name]
            builder.add_node(node.name, func)
        entry_point = self.config.entry_point
        builder.set_entry_point(entry_point)

        edges = self.config.edges  # dependency 생성
        for edge in edges:
            pair = edge.pair
            if edge.type == "always":
                builder.add_edge(pair[0], pair[1])
                
        return builder.compile()
