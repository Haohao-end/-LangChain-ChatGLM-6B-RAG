from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Any


# 自定义GLM类
class ChatGLM2(LLM):
    max_token: int = 4096
    temperature: float = 0.8
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "custom_chatglm2"

    # 定义load_model的方法
    def load_model(self, model_path=None):
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 加载模型
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()


    # 定义_call方法：进行模型的推理
    def _call(self,prompt: str, stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(self.tokenizer,
                                        prompt,
                                        history=self.history,
                                        temperature=self.temperature,
                                        top_p=self.top_p)

        if stop is not None:
            response = enforce_stop_tokens(response, stop)

        self.history = self.history + [[None, response]]
        return response

if __name__ == '__main__':
    llm = ChatGLM2()
    llm.load_model(model_path='/Users/ligang/PycharmProjects/llm/ChatGLM-6B/THUDM/chatglm-6b-int4')
    print(f'llm--->{llm}')
    print(llm("1+1等于几？"))


