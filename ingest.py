import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = 'data1/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
# EMBEDDINGS_MODEL_PATH = "models"
# os.environ['SENTENCE_TRANSFORMERS_HOME'] = EMBEDDINGS_MODEL_PATH

# Create vector database
def create_vector_db():
    # loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True, show_progress=True)
    # loader = CSVLoader(file_path="data\sys.csv")
    documents = loader.load()
    # print("length of docs", len(text_data))
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts = text_splitter.split_documents(documents)    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()
