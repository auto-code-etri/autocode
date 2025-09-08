from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
import yaml_include
from pydantic import BaseModel


class SourceConfig(BaseModel):
    name: str
    type: str
    kwargs: Optional[dict] = None


class DatasetConfig(BaseModel):
    name: str
    type: str
    remove: bool = False
    kwargs: Optional[dict] = None


class EdgeConfig(BaseModel):
    pair: Tuple[str, str]
    type: str
    kwargs: Optional[dict] = None
    
class NodeConfig(BaseModel):
    name: str
    type: str
    input_keys: List[str]
    key_map: Dict[str, str] = None
    kwargs: Optional[dict] = {}
    def __init__(self, **data):
        super().__init__(**data)

        if self.key_map is None:
            self.key_map = {key: key for key in self.input_keys}
        else:
            self.key_map = {**{key: key for key in self.input_keys}, **self.key_map}

class GraphConfig(BaseModel):
    entry_point: str
    edges: List[EdgeConfig]
    nodes: List[NodeConfig]

class Config(BaseModel):
    description: Optional[str] = None
    source: List[SourceConfig] = None
    dataset: List[DatasetConfig] = None
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
        



