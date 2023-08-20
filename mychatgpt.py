import os
import sys

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = sys.argv[1]

loader1 = TextLoader("code/code-penal-1-dz.txt")
loader2 = TextLoader("code/code-penal-2-dz.txt")
loader3 = TextLoader("code/code-penal-3-dz.txt")
loader4 = TextLoader("code/code-penal-4-dz.txt")
#loader = TextLoader("data/quran.txt")
index = VectorstoreIndexCreator().from_loaders([loader1, loader2, loader3, loader4])
#index = VectorstoreIndexCreator().from_loaders([loader])
print(index.query(query))
