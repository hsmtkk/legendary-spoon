import boto3
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
from langchain.chains.retrieval_qa.base import RetrievalQA

embeddings = OpenAIEmbeddings()

access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

service = "aoss"  # must set the service as 'aoss'
region = "ap-northeast-1"
credentials = boto3.Session(
    aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key
).get_credentials()
awsauth = AWS4Auth(
    access_key_id, secret_access_key, region, service, session_token=credentials.token
)

vector = OpenSearchVectorSearch(
    embedding_function=embeddings,
    opensearch_url=os.environ["OPENSEARCH_URL"],
    http_auth=awsauth,
    timeout=300,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    index_name="test-index-using-aoss",
    engine="faiss",
)

llm = ChatOpenAI()

retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=vector.as_retriever())

ans = retrievalQA.invoke("ワクチンのメリット、デメリットは何か")
print(ans)

ans = retrievalQA.invoke("ワクチン接種後の有害事象は何か")
print(ans)
