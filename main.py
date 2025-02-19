# coding:utf-8
# 导入必备的工具包
from langchain.prompts import PromptTemplate
from get_vector import *
from model import ChatGLM2
# 加载FAISS向量库
EMBEDDING_MODEL = './m3e-base'
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.load_local('./faiss/camp', embeddings)


def get_related_content(related_docs):
    related_content = []
    for doc in related_docs:
        related_content.append(doc.page_content.replace('\n\n', '\n'))
    return '\n'.join(related_content)

def define_prompt():
    question = '我买的商品来自于哪个仓库，从哪出发的，预计什么到达'
    docs = db.similarity_search(question, k=1)
    # print(f'docs-->{docs}')
    related_docs = get_related_content(docs)

    # 构建模板
    PROMPT_TEMPLATE = """
           基于以下已知信息，简洁和专业的来回答用户的问题。不允许在答案中添加编造成分。
           已知内容:
           {context}
           问题:
           {question}"""
    prompt = PromptTemplate(input_variables=["context", "question"],
                            template=PROMPT_TEMPLATE)

    my_prompt = prompt.format(context=related_docs,
                                question=question)
    return my_prompt

def qa():
    llm = ChatGLM2()
    llm.load_model('/Users/ligang/PycharmProjects/llm/ChatGLM-6B/THUDM/chatglm-6b')
    my_prompt = define_prompt()
    result = llm(my_prompt)
    return result

if __name__ == '__main__':
    result = qa()
    print(f'result-->{result}')
