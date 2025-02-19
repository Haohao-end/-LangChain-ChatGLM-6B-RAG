from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # 向量数据库
# from langchain.document_loaders import UnstructuredFileLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS  # 向量数据库

def main():
    # 定义向量模型路径
    EMBEDDING_MODEL = './m3e-base'

    # 第一步：加载文档：
    loader = UnstructuredFileLoader('物流信息.txt')
    data = loader.load()
    # print(f'data-->{data}')
    # 第二步：切分文档：
    text_split = RecursiveCharacterTextSplitter(chunk_size=128,
                                                chunk_overlap=4)
    split_data = text_split.split_documents(data)
    # print(f'split_data-->{split_data}')

    # 第三步：初始化huggingface模型embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 第四步：将切分后的文档进行向量化，并且存储下来
    db = FAISS.from_documents(split_data, embeddings)
    db.save_local('./faiss/camp')

    return split_data


if __name__ == '__main__':
    split_data = main()
    print(f'split_data-->{split_data}')

