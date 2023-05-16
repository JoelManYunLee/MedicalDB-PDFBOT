#import dependencies

from apikey import apikey 

import os
import argparse

from langchain.document_loaders import OnlinePDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

model_id = "gpt-3.5-turbo"

#temporarily load a PDF from some archive, need to change this to the pdf that is retrieved from pubmed API
loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
pdfData = loader.load()

#print the first page of the
print(pdfData)
