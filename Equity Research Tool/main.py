import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0.9, max_tokens=500)

file_path = 'faiss_store_openai.pkl'

st.title('News Research Tool📈')
st.sidebar.title('News Article URLs')
main_placefolder = st.empty()

#create a sidebar with inputs for url
urls = []
for i in range(3):
    url = st.sidebar.text_input(f'URL {i+1}')
    urls.append(url)

process_url_clicked = st.sidebar.button('Process URLs')

if process_url_clicked:
    #load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text('Data Loading...Started...⏳⏳⏳')
    data = loader.load()

    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','], chunk_size=1000)

    main_placefolder.text('Text Splitter...Started...⏳⏳⏳')
    docs = text_splitter.split_documents(data)

    #embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embedding=embeddings)
    main_placefolder.text('Embedding Vector Started Building...⏳⏳⏳')
    time.sleep(2)

    #save in a pickle format
    vectorindex_openai_to_bytes = vectorstore_openai.serialize_to_bytes()
    with open(file_path, 'wb') as f:
        pickle.dump(vectorindex_openai_to_bytes, f)

#load info from the file
query = main_placefolder.text_input('Question: ')
embeddings = OpenAIEmbeddings()
if query:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            vectorIndex = pickle.load(f)
            vectorindex_openai_from_bytes = FAISS.deserialize_from_bytes(
                embeddings=embeddings, serialized=vectorIndex)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorindex_openai_from_bytes.as_retriever())
            result = chain({'question': query}, return_only_outputs=True)
            st.header('Answer')
            st.write(result['answer'])
            
            #display source
            sources = result.get('sources', '')
            if sources:
                st.subheader('Sources:')
                sources_list = sources.split('\n')
                for source in sources_list:
                    st.write(source)
            
