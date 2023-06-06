from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, VectorParams, Distance,PointStruct,UpdateStatus
import os
import datetime
import uuid

QDRANT_HOST=os.getenv("QDRANT_HOST","localhost")
QDRANT_PORT=os.getenv("QDRANT_PORT",6333)
QDRANT_VECTOR_DIMENSIONALITY=os.getenv("QDRANT_VECTOR_DIMENSIONALITY",1536)
QDRANT_DEFAULT_COLLECTION_NAME =os.getenv("QDRANT_DEFAULT_COLLECTION_NAME","defaultcollection")
QDRANT_DEFAULT_WEB_COLLECTION_NAME="web"+QDRANT_DEFAULT_COLLECTION_NAME

dummy_vector = [0.0] * QDRANT_VECTOR_DIMENSIONALITY

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def does_collection_exists(collection_name: str) -> bool:
    try:
        collection_info = client.get_collection(collection_name)
        print(f'Collection "{collection_name}" already exists')
        return True
    except Exception as e:
        if isinstance(e, UnexpectedResponse) and e.status_code == 404:
            print(f'Collection "{collection_name}" does not exist')
            return False
        else:
            print(e)
            return False

# We create a default collection if it doesn't exist. 
# The default limits in Qdrant are:
# Maximum number of vectors per collection: 1 billion
# Maximum number of dimensions per collection: 32768
if not does_collection_exists(QDRANT_DEFAULT_COLLECTION_NAME):
    try:
        client.recreate_collection(collection_name=QDRANT_DEFAULT_COLLECTION_NAME, vectors_config=VectorParams(size=QDRANT_VECTOR_DIMENSIONALITY, distance=Distance.COSINE))
        print(f'Collection "{QDRANT_DEFAULT_COLLECTION_NAME}" created successfully')
    except Exception as e:
        print(e)
        
if not does_collection_exists(QDRANT_DEFAULT_WEB_COLLECTION_NAME):
    try:
        client.recreate_collection(collection_name=QDRANT_DEFAULT_WEB_COLLECTION_NAME, vectors_config=VectorParams(size=QDRANT_VECTOR_DIMENSIONALITY, distance=Distance.COSINE))
        print(f'Collection "{QDRANT_DEFAULT_WEB_COLLECTION_NAME}" created successfully')
    except Exception as e:
        print(e)

def search_vectors(vector: List[float], top_k: int,docs_tobesearched:List[str],collection_name: str=QDRANT_DEFAULT_COLLECTION_NAME) -> List[Tuple[List[float], str, str]]:
    query_filter=Filter(should=[])
    if docs_tobesearched is None:
        query_filter=None
    else:
        for doc_name in docs_tobesearched:
            query_filter.should.append(FieldCondition(key="document_name",match=MatchValue(value=doc_name)))

    search_result = client.search(
        collection_name, vector, query_filter=query_filter,limit=top_k,with_payload=True
    )

    texts=[]
    doc_names=[]
    for result in search_result:
        payload = result.payload
        texts.append(payload['text'])
        doc_names.append(payload['document_name'])

    return texts, doc_names

#Collection should be created in Qdrant before calling this function
def does_embedding_exists(doc_name: str,collection_name: str=QDRANT_DEFAULT_COLLECTION_NAME) -> bool:
    query_filter=Filter(
            must=[FieldCondition(key="document_name",match=MatchValue(value=doc_name))]
        )

    search_result = client.search(
        collection_name,
        dummy_vector,
        limit=1,
        query_filter=query_filter
    )

    return bool(search_result)


def add_to_qdrant(text_array: List[str], embeddings_array: List[List[float]], doc_name: str,collection_name: str=QDRANT_DEFAULT_COLLECTION_NAME) -> None:
    for i, (embedding, text) in enumerate(zip(embeddings_array, text_array)):
        vector_data = {
            'id': uuid.uuid4().hex, #generate new unique id for each vector here 
            'vector': embedding,
            'payload': {'text': text, 'document_name': doc_name , 'timestamp': datetime.datetime.utcnow()}, #we can use this timestamp to delete old embeddings from Qdrant later
        }
        res=client.upsert(collection_name, [vector_data],wait=True)
        if not res.status==UpdateStatus.COMPLETED:
            print(f"Embeddings for document {doc_name} failed to add to collection.")
            #Can add retry logic here
