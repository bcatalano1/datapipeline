import pyarrow.parquet as pq
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
api_key = os.getenv("NEW_RELIC_API_KEY")
pc = Pinecone(api_key=api_key)
index = pc.Index("baking-recipes")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize the chunker (targeting ~200 words per chunk with overlap)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)

# Read the Parquet file in chunks of 500 rows to preserve Docker RAM
parquet_file = pq.ParquetFile("data/processed/specialized_baking_corpus.parquet")

batch_size = 500
for batch in parquet_file.iter_batches(batch_size=batch_size):
    df = batch.to_pandas()
    upsert_data = []
    for _, row in df.iterrows():
        # Generate a stable ID for full recipe
        parent_id = str(uuid.uuid4())

        # Combine the text
        full_text = f"{row['title']}. Ingredients: {row['ingredients']}. Directions: {row['directions']}"        

        # Break the text into memory-safe chunks
        chunks = text_splitter.split_text(full_text)

        #Embed the chunks 
        embedding = model.encode(chunks)

        for i, (chunk_text, embed) in enumerate(zip(chunks, embedding)):
            # Create a unique ID for each chunk
            chunk_id = f"{parent_id}_chunk-{i}"
            upsert_data.append((
                chunk_id, 
                embed.tolist(), 
                {
                    "parent_id": parent_id, 
                    "text": chunk_text
                }
            ))
        # Upsert the flattened list of all chunks for this batch
        index.upsert(vectors=upsert_data)
        print(f"Upserted {len(upsert_data)} chunks from {len(df)} recipes.")

