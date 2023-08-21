{"filter":false,"title":"search_info.py","tooltip":"/interactive-chat-using-Lex-and-ChatGPT/cdk-chatbot/lib/search_info.py","undoManager":{"mark":41,"position":41,"stack":[[{"start":{"row":0,"column":0},"end":{"row":52,"column":0},"action":"insert","lines":["from PyPDF2 import PdfReader","import glob","import os","","class CustomDocument:","    def __init__(self, page_content, metadata):","        self.page_content = page_content","        self.metadata = metadata","","def load_pdfs_from_directory(directory, glob_pattern=\"*.pdf\"):","    documents = []","    for filepath in glob.glob(os.path.join(directory, glob_pattern)):","        with open(filepath, 'rb') as file:","            reader = PdfReader(file)","            text = \"\"","            for page_num in range(len(reader.pages)):","                page = reader.pages[page_num]","                text += page.extract_text()","            # 여기서 CustomDocument 객체를 생성","            document = CustomDocument(page_content=text, metadata={\"source\": filepath})","            documents.append(document)","    return documents","","directory_path = './pdf_files'  # 디렉터리 경로 (필요한 경우 수정)","documents = load_pdfs_from_directory(directory_path)","print(f\"Loaded {len(documents)} documents from the directory.\")","","text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)","texts = text_splitter.split_documents(documents)","print(f\"Split into {len(texts)} text chunks.\")  # 분할된 텍스트 청크의 수 확인","","# 텍스트 분리기를 사용하여 문서를 청크로 분할","texts = text_splitter.split_documents(documents)","print(f\"Split into {len(texts)} text chunks.\")  # 분할된 텍스트 청크의 수 확인","","persist_directory = 'pdf_db'  # 벡터 데이터베이스 저장 디렉터리","","embedding = OpenAIEmbeddings()  # OpenAI 임베딩 사용 설정","","# Chroma 벡터 데이터베이스 생성 및 문서 임베딩","vectordb = Chroma.from_documents(","    documents=texts,","    embedding=embedding,","    persist_directory=persist_directory)","","vectordb.persist()  # 벡터 데이터베이스 지속성 저장","vectordb = None  # 참조 제거","","# 지속성 저장된 벡터 데이터베이스 로드","vectordb = Chroma(","    persist_directory=persist_directory,","    embedding_function=embedding)",""],"id":1}],[{"start":{"row":51,"column":33},"end":{"row":52,"column":0},"action":"remove","lines":["",""],"id":2}],[{"start":{"row":3,"column":0},"end":{"row":15,"column":0},"action":"insert","lines":["import openai","from PyPDF2 import PdfReader","from langchain.text_splitter import CharacterTextSplitter","from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS","import os","from langchain.vectorstores import Chroma","from langchain.embeddings import OpenAIEmbeddings","from langchain.text_splitter import RecursiveCharacterTextSplitter","from langchain.llms import OpenAI","from langchain.chains import RetrievalQA","from langchain.document_loaders import TextLoader","from langchain.document_loaders import DirectoryLoader",""],"id":3}],[{"start":{"row":4,"column":0},"end":{"row":5,"column":0},"action":"remove","lines":["from PyPDF2 import PdfReader",""],"id":4}],[{"start":{"row":14,"column":0},"end":{"row":14,"column":84},"action":"insert","lines":["os.environ[\"OPENAI_API_KEY\"] = \"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\""],"id":5}],[{"start":{"row":62,"column":33},"end":{"row":63,"column":0},"action":"insert","lines":["",""],"id":6},{"start":{"row":63,"column":0},"end":{"row":63,"column":4},"action":"insert","lines":["    "]},{"start":{"row":63,"column":4},"end":{"row":64,"column":0},"action":"insert","lines":["",""]},{"start":{"row":64,"column":0},"end":{"row":64,"column":4},"action":"insert","lines":["    "]}],[{"start":{"row":64,"column":0},"end":{"row":64,"column":4},"action":"remove","lines":["    "],"id":7},{"start":{"row":63,"column":4},"end":{"row":64,"column":0},"action":"remove","lines":["",""]},{"start":{"row":63,"column":0},"end":{"row":63,"column":4},"action":"remove","lines":["    "]},{"start":{"row":62,"column":33},"end":{"row":63,"column":0},"action":"remove","lines":["",""]}],[{"start":{"row":62,"column":33},"end":{"row":63,"column":0},"action":"insert","lines":["",""],"id":8},{"start":{"row":63,"column":0},"end":{"row":63,"column":4},"action":"insert","lines":["    "]},{"start":{"row":63,"column":4},"end":{"row":64,"column":0},"action":"insert","lines":["",""]},{"start":{"row":64,"column":0},"end":{"row":64,"column":4},"action":"insert","lines":["    "]}],[{"start":{"row":64,"column":0},"end":{"row":64,"column":4},"action":"remove","lines":["    "],"id":9}],[{"start":{"row":64,"column":0},"end":{"row":80,"column":0},"action":"insert","lines":["import pdfplumber","","pdf_path = \"pdf_files/재해통계및사례집_2021.pdf\"","","# pdfplumber를 사용하여 PDF 파일 열기","with pdfplumber.open(pdf_path) as pdf:","    raw_text = \"\"","","    # 각 페이지에 대해 텍스트 추출","    for page in pdf.pages:","        text = page.extract_text()","        if text:","            raw_text += text","","# 추출된 텍스트의 일부분 출력","print(raw_text[:1000])",""],"id":10}],[{"start":{"row":80,"column":0},"end":{"row":81,"column":0},"action":"insert","lines":["",""],"id":11}],[{"start":{"row":0,"column":0},"end":{"row":81,"column":0},"action":"remove","lines":["from PyPDF2 import PdfReader","import glob","import os","import openai","from langchain.text_splitter import CharacterTextSplitter","from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS","import os","from langchain.vectorstores import Chroma","from langchain.embeddings import OpenAIEmbeddings","from langchain.text_splitter import RecursiveCharacterTextSplitter","from langchain.llms import OpenAI","from langchain.chains import RetrievalQA","from langchain.document_loaders import TextLoader","from langchain.document_loaders import DirectoryLoader","os.environ[\"OPENAI_API_KEY\"] = \"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\"","class CustomDocument:","    def __init__(self, page_content, metadata):","        self.page_content = page_content","        self.metadata = metadata","","def load_pdfs_from_directory(directory, glob_pattern=\"*.pdf\"):","    documents = []","    for filepath in glob.glob(os.path.join(directory, glob_pattern)):","        with open(filepath, 'rb') as file:","            reader = PdfReader(file)","            text = \"\"","            for page_num in range(len(reader.pages)):","                page = reader.pages[page_num]","                text += page.extract_text()","            # 여기서 CustomDocument 객체를 생성","            document = CustomDocument(page_content=text, metadata={\"source\": filepath})","            documents.append(document)","    return documents","","directory_path = './pdf_files'  # 디렉터리 경로 (필요한 경우 수정)","documents = load_pdfs_from_directory(directory_path)","print(f\"Loaded {len(documents)} documents from the directory.\")","","text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)","texts = text_splitter.split_documents(documents)","print(f\"Split into {len(texts)} text chunks.\")  # 분할된 텍스트 청크의 수 확인","","# 텍스트 분리기를 사용하여 문서를 청크로 분할","texts = text_splitter.split_documents(documents)","print(f\"Split into {len(texts)} text chunks.\")  # 분할된 텍스트 청크의 수 확인","","persist_directory = 'pdf_db'  # 벡터 데이터베이스 저장 디렉터리","","embedding = OpenAIEmbeddings()  # OpenAI 임베딩 사용 설정","","# Chroma 벡터 데이터베이스 생성 및 문서 임베딩","vectordb = Chroma.from_documents(","    documents=texts,","    embedding=embedding,","    persist_directory=persist_directory)","","vectordb.persist()  # 벡터 데이터베이스 지속성 저장","vectordb = None  # 참조 제거","","# 지속성 저장된 벡터 데이터베이스 로드","vectordb = Chroma(","    persist_directory=persist_directory,","    embedding_function=embedding)","    ","import pdfplumber","","pdf_path = \"pdf_files/재해통계및사례집_2021.pdf\"","","# pdfplumber를 사용하여 PDF 파일 열기","with pdfplumber.open(pdf_path) as pdf:","    raw_text = \"\"","","    # 각 페이지에 대해 텍스트 추출","    for page in pdf.pages:","        text = page.extract_text()","        if text:","            raw_text += text","","# 추출된 텍스트의 일부분 출력","print(raw_text[:1000])","",""],"id":12},{"start":{"row":0,"column":0},"end":{"row":77,"column":0},"action":"insert","lines":["# 필요한 라이브러리 및 모듈을 가져옵니다.","from PyPDF2 import PdfReader","import glob","import os","import openai","from langchain.text_splitter import CharacterTextSplitter","from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS","from langchain.vectorstores import Chroma","from langchain.embeddings import OpenAIEmbeddings","from langchain.text_splitter import RecursiveCharacterTextSplitter","from langchain.llms import OpenAI","from langchain.chains import RetrievalQA","from langchain.document_loaders import TextLoader, DirectoryLoader","","# OpenAI API 키 설정","os.environ[\"OPENAI_API_KEY\"] = \"YOUR_API_KEY\"","","# 사용자 정의 문서 클래스","class CustomDocument:","    def __init__(self, page_content, metadata):","        self.page_content = page_content","        self.metadata = metadata","","# 지정된 디렉터리에서 PDF 파일을 로드하는 함수","def load_pdfs_from_directory(directory, glob_pattern=\"*.pdf\"):","    documents = []","    for filepath in glob.glob(os.path.join(directory, glob_pattern)):","        with open(filepath, 'rb') as file:","            reader = PdfReader(file)","            text = \"\"","            for page_num in range(len(reader.pages)):","                page = reader.pages[page_num]","                text += page.extract_text()","            document = CustomDocument(page_content=text, metadata={\"source\": filepath})","            documents.append(document)","    return documents","","# PDF 파일 로드","directory_path = './pdf_files'","documents = load_pdfs_from_directory(directory_path)","print(f\"Loaded {len(documents)} documents from the directory.\")","","# 텍스트 분리기 인스턴스 생성","text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)","texts = text_splitter.split_documents(documents)","print(f\"Split into {len(texts)} text chunks.\")","","# OpenAI 임베딩 인스턴스 생성","embedding = OpenAIEmbeddings()","","# Chroma 벡터 데이터베이스 생성","persist_directory = 'pdf_db'","vectordb = Chroma.from_documents(","    documents=texts,","    embedding=embedding,","    persist_directory=persist_directory)","","vectordb.persist()","vectordb = None","","# 벡터 데이터베이스 로드","vectordb = Chroma(","    persist_directory=persist_directory,","    embedding_function=embedding)","    ","import pdfplumber","","pdf_path = \"pdf_files/재해통계및사례집_2021.pdf\"","","# pdfplumber를 사용하여 PDF 파일 열기","with pdfplumber.open(pdf_path) as pdf:","    raw_text = \"\"","    for page in pdf.pages:","        text = page.extract_text()","        if text:","            raw_text += text","print(raw_text[:1000])",""]}],[{"start":{"row":13,"column":0},"end":{"row":14,"column":0},"action":"insert","lines":["@model_validator(pre=True, skip_on_failure=True)",""],"id":13}],[{"start":{"row":13,"column":48},"end":{"row":14,"column":0},"action":"remove","lines":["",""],"id":14}],[{"start":{"row":13,"column":0},"end":{"row":14,"column":0},"action":"remove","lines":["@model_validator(pre=True, skip_on_failure=True)",""],"id":15}],[{"start":{"row":14,"column":32},"end":{"row":14,"column":44},"action":"remove","lines":["YOUR_API_KEY"],"id":16},{"start":{"row":14,"column":32},"end":{"row":14,"column":83},"action":"insert","lines":["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"]}],[{"start":{"row":7,"column":0},"end":{"row":8,"column":0},"action":"remove","lines":["from langchain.vectorstores import Chroma",""],"id":18},{"start":{"row":7,"column":0},"end":{"row":8,"column":0},"action":"insert","lines":["from langchain.vectorstores.chroma import Chroma",""]}],[{"start":{"row":12,"column":66},"end":{"row":13,"column":0},"action":"insert","lines":["",""],"id":19}],[{"start":{"row":13,"column":0},"end":{"row":14,"column":0},"action":"insert","lines":["from langchain.vectorstores import pinecone",""],"id":20}],[{"start":{"row":13,"column":0},"end":{"row":14,"column":0},"action":"remove","lines":["from langchain.vectorstores import pinecone",""],"id":22}],[{"start":{"row":6,"column":56},"end":{"row":6,"column":74},"action":"remove","lines":["Pinecone, Weaviate"],"id":23}],[{"start":{"row":6,"column":54},"end":{"row":6,"column":55},"action":"remove","lines":[","],"id":26},{"start":{"row":6,"column":54},"end":{"row":6,"column":55},"action":"remove","lines":[" "]}],[{"start":{"row":7,"column":0},"end":{"row":8,"column":0},"action":"remove","lines":["from langchain.vectorstores.chroma import Chroma",""],"id":27},{"start":{"row":7,"column":0},"end":{"row":8,"column":0},"action":"insert","lines":["from langchain.vectorstores.chroma import Chroma",""]}],[{"start":{"row":7,"column":0},"end":{"row":8,"column":0},"action":"remove","lines":["from langchain.vectorstores.chroma import Chroma",""],"id":28},{"start":{"row":7,"column":0},"end":{"row":8,"column":0},"action":"insert","lines":["from langchain.vectorstores import Chroma",""]}],[{"start":{"row":7,"column":0},"end":{"row":7,"column":1},"action":"insert","lines":["#"],"id":37}],[{"start":{"row":5,"column":0},"end":{"row":5,"column":1},"action":"insert","lines":["#"],"id":38}],[{"start":{"row":9,"column":0},"end":{"row":9,"column":1},"action":"insert","lines":["#"],"id":39}],[{"start":{"row":11,"column":0},"end":{"row":11,"column":1},"action":"insert","lines":["#"],"id":40}],[{"start":{"row":12,"column":0},"end":{"row":12,"column":1},"action":"insert","lines":["#"],"id":41}],[{"start":{"row":1,"column":0},"end":{"row":13,"column":0},"action":"remove","lines":["from PyPDF2 import PdfReader","import glob","import os","import openai","#from langchain.text_splitter import CharacterTextSplitter","from langchain.vectorstores import ElasticVectorSearch, FAISS","#from langchain.vectorstores import Chroma","from langchain.embeddings import OpenAIEmbeddings","#from langchain.text_splitter import RecursiveCharacterTextSplitter","from langchain.llms import OpenAI","#from langchain.chains import RetrievalQA","#from langchain.document_loaders import TextLoader, DirectoryLoader",""],"id":42},{"start":{"row":1,"column":0},"end":{"row":8,"column":0},"action":"insert","lines":["import openai","from PyPDF2 import PdfReader","from langchain.embeddings.openai import OpenAIEmbeddings","from langchain.text_splitter import CharacterTextSplitter","from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS","import os","",""]}],[{"start":{"row":1,"column":0},"end":{"row":7,"column":0},"action":"remove","lines":["import openai","from PyPDF2 import PdfReader","from langchain.embeddings.openai import OpenAIEmbeddings","from langchain.text_splitter import CharacterTextSplitter","from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS","import os",""],"id":43},{"start":{"row":1,"column":0},"end":{"row":7,"column":54},"action":"insert","lines":["from langchain.vectorstores import Chroma","from langchain.embeddings import OpenAIEmbeddings","from langchain.text_splitter import RecursiveCharacterTextSplitter","from langchain.llms import OpenAI","from langchain.chains import RetrievalQA","from langchain.document_loaders import TextLoader","from langchain.document_loaders import DirectoryLoader"]}],[{"start":{"row":8,"column":0},"end":{"row":8,"column":1},"action":"insert","lines":["o"],"id":44}],[{"start":{"row":8,"column":0},"end":{"row":8,"column":1},"action":"remove","lines":["o"],"id":45}],[{"start":{"row":8,"column":0},"end":{"row":8,"column":1},"action":"insert","lines":["ㅑ"],"id":46},{"start":{"row":8,"column":1},"end":{"row":8,"column":2},"action":"insert","lines":["ㅡ"]},{"start":{"row":8,"column":2},"end":{"row":8,"column":3},"action":"insert","lines":["ㅔ"]},{"start":{"row":8,"column":3},"end":{"row":8,"column":4},"action":"insert","lines":["ㅐ"]},{"start":{"row":8,"column":3},"end":{"row":8,"column":4},"action":"remove","lines":["ㅐ"]},{"start":{"row":8,"column":2},"end":{"row":8,"column":3},"action":"remove","lines":["ㅔ"]},{"start":{"row":8,"column":1},"end":{"row":8,"column":2},"action":"remove","lines":["ㅡ"]},{"start":{"row":8,"column":0},"end":{"row":8,"column":1},"action":"remove","lines":["ㅑ"]}],[{"start":{"row":8,"column":0},"end":{"row":8,"column":1},"action":"insert","lines":["I"],"id":47},{"start":{"row":8,"column":1},"end":{"row":8,"column":2},"action":"insert","lines":["m"]},{"start":{"row":8,"column":2},"end":{"row":8,"column":3},"action":"insert","lines":["p"]},{"start":{"row":8,"column":3},"end":{"row":8,"column":4},"action":"insert","lines":["o"]},{"start":{"row":8,"column":4},"end":{"row":8,"column":5},"action":"insert","lines":["r"]}],[{"start":{"row":8,"column":4},"end":{"row":8,"column":5},"action":"remove","lines":["r"],"id":48},{"start":{"row":8,"column":3},"end":{"row":8,"column":4},"action":"remove","lines":["o"]},{"start":{"row":8,"column":2},"end":{"row":8,"column":3},"action":"remove","lines":["p"]},{"start":{"row":8,"column":1},"end":{"row":8,"column":2},"action":"remove","lines":["m"]},{"start":{"row":8,"column":0},"end":{"row":8,"column":1},"action":"remove","lines":["I"]}],[{"start":{"row":8,"column":0},"end":{"row":8,"column":1},"action":"insert","lines":["i"],"id":49},{"start":{"row":8,"column":1},"end":{"row":8,"column":2},"action":"insert","lines":["m"]},{"start":{"row":8,"column":2},"end":{"row":8,"column":3},"action":"insert","lines":["p"]},{"start":{"row":8,"column":3},"end":{"row":8,"column":4},"action":"insert","lines":["o"]},{"start":{"row":8,"column":4},"end":{"row":8,"column":5},"action":"insert","lines":["r"]},{"start":{"row":8,"column":5},"end":{"row":8,"column":6},"action":"insert","lines":["t"]}],[{"start":{"row":8,"column":6},"end":{"row":8,"column":7},"action":"insert","lines":[" "],"id":50},{"start":{"row":8,"column":7},"end":{"row":8,"column":8},"action":"insert","lines":["o"]},{"start":{"row":8,"column":8},"end":{"row":8,"column":9},"action":"insert","lines":["d"]}],[{"start":{"row":8,"column":8},"end":{"row":8,"column":9},"action":"remove","lines":["d"],"id":51}],[{"start":{"row":8,"column":8},"end":{"row":8,"column":9},"action":"insert","lines":["s"],"id":52}],[{"start":{"row":8,"column":9},"end":{"row":9,"column":0},"action":"insert","lines":["",""],"id":53},{"start":{"row":9,"column":0},"end":{"row":9,"column":1},"action":"insert","lines":["i"]},{"start":{"row":9,"column":1},"end":{"row":9,"column":2},"action":"insert","lines":["m"]},{"start":{"row":9,"column":2},"end":{"row":9,"column":3},"action":"insert","lines":["p"]},{"start":{"row":9,"column":3},"end":{"row":9,"column":4},"action":"insert","lines":["o"]},{"start":{"row":9,"column":4},"end":{"row":9,"column":5},"action":"insert","lines":["r"]},{"start":{"row":9,"column":5},"end":{"row":9,"column":6},"action":"insert","lines":["t"]}],[{"start":{"row":9,"column":6},"end":{"row":9,"column":7},"action":"insert","lines":[" "],"id":54},{"start":{"row":9,"column":7},"end":{"row":9,"column":8},"action":"insert","lines":["g"]},{"start":{"row":9,"column":8},"end":{"row":9,"column":9},"action":"insert","lines":["l"]},{"start":{"row":9,"column":9},"end":{"row":9,"column":10},"action":"insert","lines":["o"]},{"start":{"row":9,"column":10},"end":{"row":9,"column":11},"action":"insert","lines":["b"]}]]},"ace":{"folds":[],"scrolltop":24.492919921875,"scrollleft":0,"selection":{"start":{"row":20,"column":59},"end":{"row":20,"column":62},"isBackwards":false},"options":{"guessTabSize":true,"useWrapMode":false,"wrapToView":true},"firstLineState":{"row":0,"state":"start","mode":"ace/mode/python"}},"timestamp":1692341987465,"hash":"98de524434f384279c3b505f0219584009b60315"}