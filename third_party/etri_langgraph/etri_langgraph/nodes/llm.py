from typing import Any, List, Optional, Dict
from etri_langgraph.utils.registry import (
    node_registry, prompt_registry, model_registry, BaseNode,
)
from langchain_core.output_parsers import StrOutputParser

@node_registry(name="llm")
class LLMNode(BaseNode):
    def __init__(
        self,
        key: str,
        examples: Optional[dict] = None,
        llm: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.key = key
        self.examples = examples
        self.llm = llm or {}
        self.kwargs = kwargs
        self.prompt_conf = kwargs.get("prompt", {})

    async def run(self, data: List[dict]) -> dict:
        data = data[-1]
        prompt_type = self.prompt_conf.get("type")
        prompt_kwargs = self.prompt_conf.get("kwargs", {})
        input_prompt = prompt_registry[prompt_type](examples=self.examples, **prompt_kwargs)
        
        chain = (
            input_prompt
            | model_registry[prompt_type](**self.llm)
            | StrOutputParser()
        )

        result = await chain.ainvoke(data)
        
        data.update({self.kwargs.get("output_key", "output"): result, 
                     'input_prompt': input_prompt.format_messages(**data)[-1].content})

        return data
