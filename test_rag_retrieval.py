import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from feast import FeatureStore

load_dotenv()

# 1. Initialize Clients and Models
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("baking-recipes")
model = SentenceTransformer("all-MiniLM-L6-v2")

#Point this to the dir contiang feature_store.yaml
store = FeatureStore(repo_path="recipe_features/feature_repo")

#2. Embed the User's Natural Language Query
query = "How do I adjust sourdough hydration and bulk fermentation for high altitude?"
query_vector = model.encode(query).tolist()

#3. Perform a Vecotor Search with Metadata Filtering
print(f"searching pincon for: {query}")
search_results = index.query(
    vector=query_vector,
    top_k=1,
    include_metadata=True,
    filter={"requires_altitude_adjustment": {"$eq": True}}
)

if not search_results['matches']:
    print("No matching recipes found.")
    exit()

#4. Extract Data from Pinecone
best_match = search_results['matches'][0]
matched_recipe_id = int(best_match['metadata']['recipe_id'])
matched_text = best_match['metadata']['text']
score = best_match['score']

print(f"\n--- Pinecone Vector Match (Score: {score:.4f}) ---")
print(f"Matched Chunk: {matched_text}")
print(f"Recipe ID: {matched_recipe_id}")

# 5. Look up the full recipe features in the Feast SQLite Online Store
features = store.get_online_features(
    features=[
        "recipe_features:title",
        "recipe_features:ingredients",
        "recipe_features:directions",
        "recipe_features:source"
    ],
    entity_rows=[{"recipe_id": matched_recipe_id}]
).to_df()

print("\n--- Feast Online Store Data ---")
print(features.to_string())