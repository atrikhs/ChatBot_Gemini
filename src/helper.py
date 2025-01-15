from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv() 

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY



def pdf_loader(data):
    loader = DirectoryLoader(data,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader
                       )
    documents = loader.load()
    return documents


def test_splitter(docs_ext):
    split_text = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunk = split_text.split_documents(docs_ext)
    return text_chunk




def emb_vect():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings