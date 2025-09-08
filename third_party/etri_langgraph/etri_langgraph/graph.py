import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
from etri_langgraph.config import GraphConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import Callable

from langgraph.graph import StateGraph
from typing import List
import copy  # deepcopy를 위해 필요

"""registry """
from etri_langgraph.utils.registry import node_registry
from etri_langgraph.nodes import *

class Graph(BaseModel):
    config: GraphConfig
    examples: dict = {}
    etc_datasets: dict = {}

    def __init__(self, **data):
        super().__init__(**data)
    
    def run(self):
        builder = StateGraph(List[dict])    #langgraph
        nodes = self.config.nodes    #node 설정
        for node in nodes:      #initialize, excute
            func = node_registry[node.type](
                key=node.name,
                input_keys=node.input_keys,
                examples=self.examples,
                **node.kwargs,
            )
            builder.add_node(node.name, func)
        entry_point = self.config.entry_point
        builder.set_entry_point(entry_point)

        edges = self.config.edges                       #dependency 생성
        for edge in edges:
            pair = edge.pair
            if edge.type == "always":
                builder.add_edge(pair[0], pair[1])
                
        return builder.compile()