from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import test_splitter, pdf_loader, emb_vect
import os


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
print(f'{PINECONE_API_KEY = }')


os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY


pc = Pinecone(api_key=PINECONE_API_KEY)

docs_ext = pdf_loader(data = 'Data/')


chunks = test_splitter(docs_ext)

emb = emb_vect()


index_name = "chatbot"

pc.create_index(
    name=index_name,
    dimension= 768, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)



docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name = index_name,
    embedding=emb

)

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding=emb

) 
