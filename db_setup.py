import os 
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
load_dotenv()
api_key = os.getenv("NEW_RELIC_API_KEY")
pc = Pinecone(api_key=api_key)
pc.create_index(
    name="baking-recipes",
    dimension=384,
    metric="cosine",
    serverless=ServerlessSpec(cloud="aws", region="us-east-1")
)