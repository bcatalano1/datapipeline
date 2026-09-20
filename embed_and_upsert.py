import pyarrow.dataset as ds
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index = pc.Index("baking-recipes")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize the chunker (targeting ~200 words per chunk with overlap)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)

# Use pyarrow.dataset to read the partitioned Parquet directory
dataset = ds.dataset("data/processed/specialized_baking_corpus.parquet", format="parquet")

batch_size = 500
for batch in dataset.to_batches(batch_size=batch_size):
    df = batch.to_pandas()
    upsert_data = []
    for _, row in df.iterrows():
        # Use the PySpark ID to maintain the link to your Feast feature store
        recipe_id = row['recipe_id']
        parent_id_str = str(recipe_id)

        # Combine the text
        full_text = f"{row['title']}. Ingredients: {row['ingredients']}. Directions: {row['directions']}"        

        # Break the text into memory-safe chunks
        chunks = text_splitter.split_text(full_text)

        #Embed the chunks 
        embedding = model.encode(chunks)

        for i, (chunk_text, embed) in enumerate(zip(chunks, embedding)):
            # Create a unique ID for each chunk
            chunk_id = f"{parent_id_str}_chunk-{i}"
            upsert_data.append((
                chunk_id, 
                embed.tolist(), 
                {
                    "recipe_id": recipe_id,
                    "text": chunk_text,
                    # Add the PySpark flag to Pinecone metadata
                    "requires_altitude_adjustment": bool(row['requires_altitude_adjustment'])
                }
            ))
    # Upsert the flattened list of all chunks for this batch
    index.upsert(vectors=upsert_data)
    print(f"Upserted {len(upsert_data)} chunks from {len(df)} recipes.")

