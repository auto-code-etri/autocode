import re
from typing import Any, List, Optional, Dict

from etri_langgraph.utils.registry import node_registry, BaseNode


@node_registry(name="parser")
class JsonParser(BaseNode):
    def __init__(
        self,
        key: str,
        examples: Optional[dict] = None,
        **kwargs,
    ):
        self.key = key
        self.examples = examples
        self.kwargs = kwargs

    async def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self.kwargs.get("input_keys", [])[-1]
        output_key = self.kwargs.get("output_key", "parsed_output")

        raw_input = data.get(input_key, "")

        # preprocess logic inline
        code_block_pattern = r"```[a-z]*\n(.*?)```"
        matches = re.findall(code_block_pattern, raw_input, re.DOTALL)

        if matches:
            parsed_result = matches[-1]
        else:
            incomplete_pattern = r"```[a-z]*\n(.*?)$"
            fallback_matches = re.findall(incomplete_pattern, raw_input, re.DOTALL)
            if fallback_matches:
                parsed_result = fallback_matches[-1]
            else:
                parsed_result = raw_input

        data[output_key] = parsed_result
        return data

