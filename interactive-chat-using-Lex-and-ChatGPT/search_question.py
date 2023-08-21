import os
import openai
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# search_info.py 파일에서 필요한 객체와 함수를 가져옵니다.
from search_info import vectordb

# OpenAI API 키 설정
openai.api_key = "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"

# 벡터 데이터베이스를 이용해 retriever 객체 생성
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 모델 - GPT 3.5 Turbo 선택
model = "gpt-3.5-turbo"
ambiguous_responses = ["I don't know.", "모르겠어요.", "모르겠습니다.", "확인할 수 없습니다."]

def process_llm_response(llm_response):
    if any(response in llm_response['result'] for response in ambiguous_responses):
        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a well-informed system of port information."},
                {"role": "user", "content": query}
            ]
        )
        answer = response['choices'][0]['message']['content']
        print(answer)
    else: 
        print(llm_response['result'])
        
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# 질문 작성하기
query = "항만이란?"
llm_response = qa_chain(query)
process_llm_response(llm_response)
