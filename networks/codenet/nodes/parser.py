from src.utils.registry import node_registry, BaseNode

import re

@node_registry(name="jsonparser") # the upper-case key is not allowed, same as the class name but MUST be lower-case
class JsonParser(BaseNode):
    def __init__(self, **data):
        super().__init__(**data)
    
    def __call__(self, state: dict) -> dict:
        print(self.get_name())
        
        raw_input = state['llm_jun_out']

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

        state['parser_jun_out'] = parsed_result
        return state

