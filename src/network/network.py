from pydantic import BaseModel
from src.network.config import Config
from src.network.graph import Graph
from src.utils import registry

class Network(BaseModel):
    graph: Graph = None
    nx_nodes: dict = None
    nx_edges: dict = None
    
    def __init__(self, **data):
        super().__init__(**data)
    
    # 그래프 구성을 위한 정보 수집
    def gather_graph_info(self, graph_path):
        graph_config = Config(path=graph_path)
        nodes = graph_config.graph.nodes
        node_functions = {}
        for node in nodes:
            nodekey = node.name.lower()  # Assuming the registry uses lowercase names as keys
            entry = {node.name : registry.node_registry[nodekey](name=node.name)}
            node_functions.update(entry)        
        return graph_config, node_functions
    
    # 그래프 구성 및 컴파일
    def compose_and_compile(self, graph, node_functions):
        # 그래프 구성 및 컴파일
        self.graph = Graph(config=graph).compose_and_compile(node_functions=node_functions)
        
        # 네트워크의 노드 및 엣지 정보 저장
        self.nx_nodes = list(self.graph.get_graph().nodes)
        self.nx_edges = list(self.graph.get_graph().edges)
        
        return self.graph
    
    # 그래프 실행
    def run(self, state: dict):
        result = self.graph.invoke(state)
        return result
    
    # 그래프 노드 및 엣지 정보 획득
    def get_nodes_edges(self) -> dict:
        result = {}        
        result['nodes'] = self.nx_nodes
        result['edges'] = self.nx_edges
        return result
         
    # 특정 노드의 선행 노드 획득
    def get_preds(self, node_name: str):
        preds = [source for (source, target, data, conditional) in self.nx_edges if target == node_name]
        return preds
    
    # 특정 노드의 후행 노드 획득
    def get_succs(self, node_name: str):
        succs = [target for (source, target, data, conditional) in self.nx_edges if source == node_name]
        return succs