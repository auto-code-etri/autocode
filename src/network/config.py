from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
import yaml_include
from pydantic import BaseModel

class EdgeConfig(BaseModel):
    pair: Tuple[str, str]
    type: str
    kwargs: Optional[dict] = None
    
class NodeConfig(BaseModel):
    name: str
    def __init__(self, **data):
        super().__init__(**data)

class GraphConfig(BaseModel):
    entry_point: str
    nodes: List[NodeConfig]
    edges: List[EdgeConfig]

class Config(BaseModel):
    graph: GraphConfig = None

    def __init__(self, **data):
        path = data.get("path")
        if path is not None:
            yaml.add_constructor("!inc", yaml_include.Constructor())
            with open(path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            del data["path"]
            data = {**config, **data}
        super().__init__(**data)
