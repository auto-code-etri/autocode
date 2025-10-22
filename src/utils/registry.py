from typing import Callable
from typing import TypedDict, Callable, Dict, Type
from autoregistry import Registry

node_registry = Registry()

class BaseNode:
    name: str
    
    def __init__(self, **data):        
        self.name = data.get('name', '')

    def get_name(self):
        return self.name