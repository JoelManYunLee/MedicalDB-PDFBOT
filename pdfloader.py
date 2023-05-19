#import dependencies

from apikey import apikey 

import os

from langchain.document_loaders import OnlinePDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = apikey
model_id = "gpt-3.5-turbo"

class PDF_Loader:
    def loadPDF(url):
        # TODO: Temporarily load a PDF from some archive, 
        # need to change this to the pdf that is retrieved from pubmed API
        # loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
        loader = OnlinePDFLoader(url)
        pdf_data = loader.load()

        # Split up the document to avoid token threshold
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2000,
            chunk_overlap = 200,
        )

        # Create OpenAIEmbeddings and FAISS objects. Vectorize the chunks created above and save.
        documents = text_splitter.split_documents(pdf_data)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_documents(documents, embeddings)

        return docsearch

    def queryPDF(query, docsearch):
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        # pass some query to the API to interact with the PDF
        docs = docsearch.similarity_search(query)
        chain.run(input_documents=docs, question=query)


