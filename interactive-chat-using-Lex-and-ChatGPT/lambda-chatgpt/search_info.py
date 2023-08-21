# 필요한 라이브러리 및 모듈을 가져옵니다.
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
import os
import glob
from PyPDF2 import PdfReader

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "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"

# 사용자 정의 문서 클래스
class CustomDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# 지정된 디렉터리에서 PDF 파일을 로드하는 함수
def load_pdfs_from_directory(directory, glob_pattern="*.pdf"):
    documents = []
    for filepath in glob.glob(os.path.join(directory, glob_pattern)):
        with open(filepath, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            document = CustomDocument(page_content=text, metadata={"source": filepath})
            documents.append(document)
    return documents

# PDF 파일 로드
directory_path = './pdf_files'
documents = load_pdfs_from_directory(directory_path)
print(f"Loaded {len(documents)} documents from the directory.")

# 텍스트 분리기 인스턴스 생성
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} text chunks.")

# OpenAI 임베딩 인스턴스 생성
embedding = OpenAIEmbeddings()

# Chroma 벡터 데이터베이스 생성
persist_directory = 'pdf_db'
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory)

vectordb.persist()
vectordb = None

# 벡터 데이터베이스 로드
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding)
    
import pdfplumber

pdf_path = "pdf_files/재해통계및사례집_2021.pdf"

# pdfplumber를 사용하여 PDF 파일 열기
with pdfplumber.open(pdf_path) as pdf:
    raw_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            raw_text += text
print(raw_text[:1000])
