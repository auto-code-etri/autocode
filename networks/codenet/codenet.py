from src.network.network import Network
from networks.codenet.nodes import *

class CodeNet(Network):
    network_name: str = "CodeNet"
    relative_path: str = "networks/codenet"
    
    def __init__(self, **data):
        super().__init__(**data)

    # 그래프 구성 및 컴파일
    def compile(self):
        yaml_graph_path = f"{self.relative_path}/{self.network_name}.yaml"
        
        # yaml 파일에서 그래프 설정 및 노드 함수 수집
        graph_config, node_functions = self.gather_graph_info(yaml_graph_path)
               
        # 그래프 구성을 기반으로 그래프를 실제 생성하고 컴파일
        self.compose_and_compile(graph=graph_config.graph, node_functions=node_functions)
        return self.graph