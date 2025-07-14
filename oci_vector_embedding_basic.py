import sys
import json
import oracledb
import time
import uuid
import io
import locale
from langchain_core.documents import Document
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.document_loaders.oracleai import OracleTextSplitter
from langchain_community.document_loaders.oracleai import OracleDocLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import oraclevs

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

# ======== parameter 
# with open("db_config.json", "r", encoding="utf-8") as config_file:
#     config = json.load(config_file)

# with open("embedding_parameter.json", "r", encoding="utf-8") as param_file:
#     splitter_params = json.load(param_file)

# username = config["DATABASE"]["USERNAME"]
# password = config["DATABASE"]["PASSWORD"]
# host = config["DATABASE"]["HOST"]
# port = config["DATABASE"]["PORT"]
# service_name = config["DATABASE"]["SERVICE_NAME"]
# table_name = config["DATABASE"]["TABLE_NAME_CV_LANG"]
# compartment_id = config["OCI"]["compartment_id"]
# dsn = host + ":" + port + "/" + service_name

# Connection Database
# try:
#     oracledb.init_oracle_client()
#     connection = oracledb.connect(user=username, password=password, dsn=dsn)
#     print("\nOracle DB Connection successful!\n")
# except Exception as e:
#     print(f"Oracle DB Connection failed: {e}")
#     sys.exit(1)

# Oracle Langchain lib initialization
# embedder = OCIGenAIEmbeddings(
#     auth_type="INSTANCE_PRINCIPAL",
#     model_id="cohere.embed-multilingual-v3.0",
#     service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
#     compartment_id=compartment_id
# )

# splitter = OracleTextSplitter(conn=connection, params=splitter_params)
# distance_strategy = DistanceStrategy.COSINE
# table_name_with_strategy = table_name + '_' + distance_strategy

def upload_pdf_to_oracle(pdf_file_path, connection, embedder, splitter, table_name_with_strategy, distance_strategy):
    """
    PDF 파일을 Oracle 벡터 스토어에 업로드하는 함수
    """
    print(f"PDF 파일 업로드 시작: {pdf_file_path}")
    s1time = time.time()
    
    chunks_with_mdata = []
    doc_origin = Document
    max_length_oracle_allow = 9000
    counter = 0  
    document_num = 0
    
    # Oracle DocsLoader - Start
    loader_params = {}        
    loader_params['file'] = pdf_file_path
    
    # instantiate loader, splitter and embedder
    loader = OracleDocLoader(conn=connection, params=loader_params)
    
    # read the docs, convert blob docs to clob docs
    docs = loader.load()
    print(f"Number of docs loaded: {len(docs)}")

    for id, doc in enumerate(docs, start=1):
        # remove line break from the text document
        doc.page_content = doc.page_content.replace("\n", "")
        doc_origin.page_content = doc.page_content
        
        # check the doc
        if len(doc.page_content) > max_length_oracle_allow:
            # reduce the text to max_length_oracle_allow
            doc.page_content = doc.page_content[:9000]
        
        document_num += 1
        
        # chunk the doc
        chunks = splitter.split_text(doc_origin.page_content)
        print(f"Doc {id}: chunks# {len(chunks)}")

    # For each chunk create chunk_metadata with 
    for ic, chunk in enumerate(chunks, start=1):
        counter += 1  
        chunk_metadata = doc.metadata.copy()  
        chunk_metadata['id'] = str(counter)  
        chunk_metadata['document_id'] = str(document_num)
        chunks_with_mdata.append(Document(page_content=str(chunk), metadata=chunk_metadata))
        print(f"Doc {id}: metadata: {doc.metadata}")
    
    # Oracle DocsLoader - End
    
    # Count number of documents
    unique_files = set()
    for chunk in chunks_with_mdata:
        file_name = chunk.metadata['_file']
        unique_files.add(file_name)

    print("chunks_with_mdata:", chunks_with_mdata)

    vector_store = OracleVS.from_documents(
        chunks_with_mdata, 
        embedder, 
        client=connection, 
        table_name=table_name_with_strategy, 
        distance_strategy=distance_strategy
    )
    
    if vector_store is not None:
        print("\nDocuments loading, chunking and generating embeddings are complete.\n")
        result = "Documents loading, chunking and generating embeddings are complete."
    else:
        print("\nFailed to get the VectorStore populated.\n")
        result = "Failed to get the VectorStore populated."
        
    # Create Oracle HNSW Index
    oraclevs.create_index(client=connection, vector_store=vector_store, params={
        "idx_name": "hnsw" + table_name_with_strategy, "idx_type": "HNSW"
    })

    s2time = time.time()
    print(f"Vectorizing and inserting chunks duration: {round(s2time - s1time, 1)} sec.")

    return result

def safe_input(prompt):
    """안전한 입력 처리를 위한 함수"""
    try:
        return input(prompt)
    except UnicodeDecodeError:
        # 인코딩 오류 발생 시 영어로 재시도
        print("입력 인코딩 오류가 발생했습니다. 영어로 입력해주세요.")
        return input("Please enter PDF file path (type 'quit' to exit): ")

def main():
    if len(sys.argv) != 4:
        print("사용법: python upload_pdf_to_oracle.py <db_config.json> <embedding_parameter.json> <pdf_file_path>")
        sys.exit(1)

    db_config_path = sys.argv[1]
    embedding_param_path = sys.argv[2]
    pdf_path = sys.argv[3]

    # 설정 파일 읽기
    with open(db_config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    with open(embedding_param_path, "r", encoding="utf-8") as param_file:
        splitter_params = json.load(param_file)

    username = config["DATABASE"]["USERNAME"]
    password = config["DATABASE"]["PASSWORD"]
    host = config["DATABASE"]["HOST"]
    port = config["DATABASE"]["PORT"]
    service_name = config["DATABASE"]["SERVICE_NAME"]
    table_name = config["DATABASE"]["TABLE_NAME_CV_LANG"]
    compartment_id = config["OCI"]["compartment_id"]
    dsn = host + ":" + port + "/" + service_name

    # DB 연결
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

    # PDF 업로드 실행
    try:
        print(f"\nPDF 파일을 Oracle 벡터 스토어에 업로드 중입니다...")
        result = upload_pdf_to_oracle(pdf_path, connection, embedder, splitter, table_name_with_strategy, distance_strategy)
        print(f"\n결과: {result}")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
