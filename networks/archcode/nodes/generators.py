from networks.archcode.utils.parser import CodeBlockParser, DistributeParser, FindParser
from src.utils.registry import node_registry, BaseNode
from src.prompt import *
from src.model import *

from langchain_core.output_parsers import StrOutputParser
import json


@node_registry(name="requirementgenerator")
class RequirementGenerator(BaseNode):
    """
    Generate requirements based on the given user input prompt.
    inputs:
        - prompt: str
    outputs:
        - llm_jun_out: list[str]
        - input_prompt: str
    """
    def __init__(self, **data):
        super().__init__(**data)      
    
    def __call__(self, state: dict) -> dict:
        print(self.get_name())
        
        prompt_kwargs = {'body_template_paths': ['templates/prompt/archcode/requirements']}
        example = json.load(open('templates/example/archcode.json'))
        input_prompt = chat_prompt(examples=example, **prompt_kwargs)
        
        llm_params = {'max_tokens': 2048, 'model': 'gpt-4o-2024-11-20', 'platform': 'openai', 'temperature': 0, 'top_p': 1}
        chain = (
            input_prompt
            | GeneralChatModel(**llm_params)
            | CodeBlockParser()
        )

        input_data = state['input_data']
        data = {'prompt': input_data}
        
        result = chain.invoke(data)
               
        state['requirements'] = result
        state['input_prompt'] = input_prompt.format_messages(**data)[-1].content

        return state


@node_registry(name="plangenerator")
class PlanGenerator(BaseNode):
    def __init__(self, **data):
        super().__init__(**data)      
    
    def __call__(self, state: dict) -> dict:
        print(self.get_name())
        
        prompt_kwargs = {'body_template_paths': ['templates/prompt/archcode/requirements', 'templates/prompt/archcode/cot_plan']}
        example = json.load(open('templates/example/archcode.json'))
        input_prompt = chat_prompt(examples=example, **prompt_kwargs)

        llm_params = {'max_tokens': 2048, 'model': 'gpt-4o-2024-11-20', 'platform': 'openai', 'temperature': 0, 'top_p': 1}
        chain = (
            input_prompt
            | GeneralChatModel(**llm_params)
            | CodeBlockParser()
        )

        prompt = state['input_prompt']
        requirements = state['requirements']
        data = {'prompt': prompt, 'requirements': requirements}
        
        result = chain.invoke(data)
               
        state['plan'] = result
        # state['input_prompt'] = input_prompt.format_messages(**data)[-1].content

        return state


@node_registry(name="codegenerator") # the upper-case key is not allowed, same as the class name but MUST be lower-case
class CodeGenerator(BaseNode):
    def __init__(self, **data):
        super().__init__(**data)      
    
    def __call__(self, state: dict) -> dict:
        print(self.get_name())
        
        prompt_kwargs = {'body_template_paths': ['templates/prompt/archcode/requirements', 'templates/prompt/archcode/cot_plan', 'templates/prompt/archcode/cot_code']}
        example = json.load(open('templates/example/archcode.json'))
        input_prompt = chat_prompt(examples=example, **prompt_kwargs)
        
        llm_params = {'max_tokens': 2048, 'model': 'gpt-4o-2024-11-20', 'platform': 'openai', 'temperature': 0.8, 'top_p': 0.95}
        chain = (
            input_prompt
            | GeneralChatModel(**llm_params)
            | CodeBlockParser()
        )

        prompt = state['input_prompt']
        requirements = state['requirements']
        plan = state['plan']
        data = {'prompt': prompt, 'requirements': requirements, 'plan': plan}
        
        inputs = [data] * state['code_n']
        
        # Parallelize using batch
        codes = chain.batch(inputs)
        
        state['codes'] = codes
        
        return state

@node_registry(name="functestcasegenerator")
class FuncTestCaseGenerator(BaseNode):
    def __init__(self, **data):
        super().__init__(**data)      
    
    def __call__(self, state: dict) -> dict:
        print(self.get_name())
        
        prompt_kwargs = {'body_template_paths': ['templates/prompt/archcode/requirements', 'templates/prompt/archcode/func_testcases']}
        example = json.load(open('templates/example/archcode.json'))
        input_prompt = chat_prompt(examples=example, **prompt_kwargs)
        
        llm_params = {'max_tokens': 2048, 'model': 'gpt-4o-2024-11-20', 'platform': 'openai', 'temperature': 0, 'top_p': 1}
        chain = (
            input_prompt
            | GeneralChatModel(**llm_params)
            | CodeBlockParser()
            | DistributeParser(
                landmarks=[
                    ["general", "## General Cases"],
                    ["edge", "## Edge Cases"]
                ]
            )
        )

        prompt = state['input_prompt']
        requirements = state['requirements']
        data = {'prompt': prompt, 'requirements': requirements}
        
        result = chain.invoke(data)
               
        state['func_testcase'] = result
        # state['input_prompt'] = input_prompt.format_messages(**data)[-1].content

        return state

@node_registry(name="nonfunctestcasegenerator")
class NonFuncTestCaseGenerator(BaseNode):
    def __init__(self, **data):
        super().__init__(**data)      
    
    def __call__(self, state: dict) -> dict:
        print(self.get_name())
        
        prompt_kwargs = {'body_template_paths': ['templates/prompt/archcode/requirements', 'templates/prompt/archcode/nonfunc_testcases']}
        example = json.load(open('templates/example/archcode.json'))
        input_prompt = chat_prompt(examples=example, **prompt_kwargs)
        
        llm_params = {'max_tokens': 2048, 'model': 'gpt-4o-2024-11-20', 'platform': 'openai', 'temperature': 0, 'top_p': 1}
        chain = (
            input_prompt
            | GeneralChatModel(**llm_params)
            | CodeBlockParser()
            | DistributeParser(
                landmarks=[
                    # ["nfr", "# Test Cases Regarding Non-functional Requirements"],
                    ["performance", "## Performance Requirements"],
                    ["sqr", "## Specific Quality Requirements"],
                    ["robustness", "### Robustness"],
                    ["maintainability", "### Maintainability"]
                ]
            )
        )

        prompt = state['input_prompt']
        requirements = state['requirements']
        data = {'prompt': prompt, 'requirements': requirements}
        
        result = chain.invoke(data)
        
        state['nonfunc_testcase'] = result
        # state['input_prompt'] = input_prompt.format_messages(**data)[-1].content

        return state
