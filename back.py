from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community .vectorstores import FAISS
#step 1: Load the PDF file,store into documents
file_path = os.path.join("documents", "tcs.pdf")
loader = PyPDFLoader(file_path)
documents = loader.load()
print(len(documents))
#step2: convert the documnets into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
   # set a really small chunk size, just to show.
    chunk_size = 2000,
    chunk_overlap = 500
)
# let us load load chunks
mychunks = text_splitter.split_documents(documents)
print('total chunks:',len(mychunks))

#step3: create embedding model
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

#step4: create faiss database
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(embedding_function = embeddings,
                     index = index,
                     docstore = InMemoryDocstore(),
                     index_to_docstore_id={})

#step5: store our chunks into vector DB
vector_store.add_documents(mychunks)
print("sucessfully created")
#step6: store vectore store [db] permanently
vector_store.save_local("tcs_doc_index")
print("successfully created in locally")
