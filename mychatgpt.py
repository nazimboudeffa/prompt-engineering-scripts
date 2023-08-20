import os
import sys

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = sys.argv[1]

loader1 = TextLoader("data/data1.txt")
loader2 = TextLoader("data/data2.txt")
index = VectorstoreIndexCreator().from_loaders([loader1, loader2])

print(index.query(query))
