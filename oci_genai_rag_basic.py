import sys
import configparser
import oracledb
import io
import locale
import json
import argparse
from langchain_community.llms import OCIGenAI 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.document_loaders.oracleai import OracleTextSplitter
from langchain_community.document_loaders.oracleai import OracleDocLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

# UTF-8 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# 로케일 설정
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass

def load_configs(db_config_path, llm_config_path):
    """설정 파일들을 로드하는 함수"""
    # 데이터베이스 설정 로드
    with open(db_config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    # LLM 파라미터 설정 로드
    with open(llm_config_path, "r", encoding="utf-8") as llm_config_file:
        llm_config = json.load(llm_config_file)

    return config, llm_config

def parse_arguments():
    """명령행 인자를 파싱하는 함수"""
    parser = argparse.ArgumentParser(description='PDF AI Search Application with Oracle Vector Store')
    parser.add_argument('--db-config', default='db_config.json', 
                       help='데이터베이스 설정 파일 경로 (기본값: db_config.json)')
    parser.add_argument('--llm-config', default='llm_parameter_config.json', 
                       help='LLM 파라미터 설정 파일 경로 (기본값: llm_parameter_config.json)')
    parser.add_argument('question', help='질문')
    return parser.parse_args()

# 명령행 인자 파싱
args = parse_arguments()

# 설정 파일 로드
config, llm_config = load_configs(args.db_config, args.llm_config)

# 데이터베이스 설정 추출
username = config["DATABASE"]["USERNAME"]
password = config["DATABASE"]["PASSWORD"]
host = config["DATABASE"]["HOST"]
port = config["DATABASE"]["PORT"]
service_name = config["DATABASE"]["SERVICE_NAME"]
table_name = config["DATABASE"]["TABLE_NAME_CV_LANG"]
compartment_id = config["OCI"]["compartment_id"]
dsn = host + ":" + port + "/" + service_name

# LLM 파라미터 설정 추출
splitter_params = llm_config["splitter_params"]
llm_params = llm_config["llm_params"]
retriever_params = llm_config["retriever_params"]

# Connection Database
try:
    oracledb.init_oracle_client()
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    print("\nOracle DB Connection successful!\n")
except Exception as e:
    print(f"Oracle DB Connection failed: {e}")
    sys.exit(1)

# Oracle Langchain lib initialization
embedder = OCIGenAIEmbeddings(
    auth_type="INSTANCE_PRINCIPAL",
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
    compartment_id=compartment_id
)

splitter = OracleTextSplitter(conn=connection, params=splitter_params)
distance_strategy = DistanceStrategy.COSINE
table_name_with_strategy = table_name + '_' + distance_strategy
print("Table Name:", table_name_with_strategy)

# Vector Store Initialization
vector_store = OracleVS(client=connection, embedding_function=embedder, table_name=table_name_with_strategy)

def generate_response(input_text): 
    # LLM 설정
    llm = ChatOCIGenAI(
        auth_type="INSTANCE_PRINCIPAL",
        model_id="cohere.command-a-03-2025",
        service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
        compartment_id=compartment_id,
        model_kwargs=llm_params
    )  
    
    # RAG 체인 생성
    message = [
        (
            "system",
            """
            질문-답변 업무를 돕는 AI 어시스턴트입니다. 
            출처에 대한 정보를 반드시 명시해 주세요.
            문서의 내용을 참고해서 답변해 주세요.:
            \n\n
            {context}",
            """
        ),
        ("human", "{human}"),
    ]
    
    prompt = ChatPromptTemplate.from_messages(message)
    
    chain = {
        "context": vector_store.as_retriever(search_kwargs=retriever_params),
        "human": RunnablePassthrough(),
    } | prompt | llm | StrOutputParser()
    
    # 질문에 대한 답변 생성
    response = chain.invoke(input_text)
    
    return response



def main():
    print("🦜🔗 PDF AI Search Application with Oracle Vector Store")
    print("=" * 60)
    print(f"데이터베이스 설정 파일: {args.db_config}")
    print(f"LLM 파라미터 설정 파일: {args.llm_config}")
    print("=" * 60)
    
    try:
        print(f"\n질문: {args.question}")
        print("\n답변을 생성 중입니다...")
        response = generate_response(args.question)
        print(f"\n답변: {response}")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
