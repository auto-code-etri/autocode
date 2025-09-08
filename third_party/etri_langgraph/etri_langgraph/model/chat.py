import logging
import os
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult
from langchain_community.chat_models import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from etri_langgraph.utils.registry import model_registry

@model_registry(name="chat")
class GeneralChatModel(BaseChatModel):
    model: Optional[str] = None
    max_tokens: int
    temperature: float
    top_p: float
    num_ctx: Optional[int] = None
    max_retries: int = 10000
    platform: str = "azure"
    stop: Optional[List[str]] = None
    base_url: Optional[str] = None

    llm: BaseChatModel = None

    @property
    def _llm_type(self) -> str:
        return self.llm._llm_type

    @property
    def llm(self):
        if self.platform == "openai":
            return ChatOpenAI(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model_kwargs={"top_p": self.top_p},
                max_retries=self.max_retries,
            )

        elif self.platform == "vllm":
            return ChatOpenAI(
                model=self.model,
                max_tokens=self.max_tokens,
                openai_api_key="EMPTY",
                openai_api_base=os.environ['OPEN_WEBUI_BASE_URL'],
                temperature=self.temperature,
                model_kwargs={"top_p": self.top_p}
            )

        elif self.platform == "ollama":
            return ChatOllama(
                model=self.model,
                num_predict=self.max_tokens,
                num_ctx=self.num_ctx,
                temperature=self.temperature,
                top_p=self.top_p,
                base_url=self.base_url or os.environ["OLLAMA_BASE_URL"],
                headers={
                    "Content-Type": "application/json",
                },
            )

        else:
            raise ValueError(f"platform {self.platform} not supported")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[str] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            result = self.llm._generate(
                messages=messages,
                stop=stop + self.stop if stop is not None else self.stop,
                run_manager=run_manager,
                **kwargs,
            )
            return result
        except ValueError as e:
            if "content filter" in str(e):
                logging.error(f"content filter triggered")
                raise e
            elif "out of memory" in str(e):
                logging.error(f"out of memory")
                logging.error("retrying...")
                self.llm._generate(
                    messages=messages,
                    stop=stop + self.stop if stop is not None else self.stop,
                    run_manager=run_manager,
                    **kwargs,
                )
            else:
                raise e