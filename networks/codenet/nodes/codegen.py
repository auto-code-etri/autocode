from src.utils.registry import node_registry, BaseNode

from langchain_core.output_parsers import StrOutputParser
from src.prompt import *
from src.model import *

@node_registry(name="codegenerator") # the upper-case key is not allowed, same as the class name but MUST be lower-case
class CodeGenerator(BaseNode):
    def __init__(self, **data):
        super().__init__(**data)      
    
    def __call__(self, state: dict) -> dict:
        print(self.get_name())
        
        # 체인 구성을 위한 프롬프트 준비
        prompt_kwargs = {'body_template_paths': ['templates/prompt/DP']}
        input_prompt = chat_prompt(examples=None, **prompt_kwargs)
        
        # LLM 파라미터 설정 및 LLM 체인 구성
        llm_params = {'max_tokens': 4096, 'model': 'gpt-4o-2024-11-20', 'platform': 'openai', 'temperature': 0, 'top_p': 1}
        chain = (
            input_prompt
            | GeneralChatModel(**llm_params)
            | StrOutputParser()
        )

        data = state['input_data']  # input_data 가져오기
        
        result = chain.invoke(data) # 체인 실행
        
        # 결과를 state에 추가        
        state['llm_jun_out'] = result
        state['input_prompt'] = input_prompt.format_messages(**data)[-1].content

        return state
