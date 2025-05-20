import streamlit as st
import json
import os
import numpy as np
from azure.storage.blob import BlobServiceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Azure Blob Storage setup
CONN_STR = os.getenv("AZURE_CONN_STR")
CONTAINER_NAME = "user-embedding"
BLOB_NAME = "user_embeddings.json"

blob_service_client = BlobServiceClient.from_connection_string(CONN_STR)
blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)

# Load users from blob
def load_users_from_blob():
    blob_data = blob_client.download_blob().readall()
    return json.loads(blob_data)

# Add user to blob
def add_user_to_blob(user_entry):
    users = load_users_from_blob()
    users.append(user_entry)
    blob_client.upload_blob(json.dumps(users), overwrite=True)

# Build sentence from profile
def build_sentence(row):
    designation = "an intern" if int(row['intern_or_fte']) == 1 else "a Full-time employee"
    return (
        f"This person enjoys the sports: {row['fav_sports']}, "
        f"likes the music genres: {row['fav_music_genres']}, "
        f"has the hobbies: {row['hobbies']}, "
        f"liked the movies/series: {row['fav_movies_or_series']}, "
        f"knows the languages: {row['known_languages']}, "
        f"has tech interests: {row['tech_interests']}, "
        f"and is {designation} in {row['role']} in {row['division_team']} "
        f"at building {row['building_number']}, "
        f"and likes to eat {row['fav_foods']}."
    )
 
# Get embedding
def get_embedding(text):
    return model.encode(text).tolist()

# Find top 5 matches
def find_top10_matches(user_query, all_users):
    query_emb = get_embedding(user_query)
    embeddings = [user['embedding'] for user in all_users]
    similarities = cosine_similarity([query_emb], embeddings)[0]
    top10_idx = np.argsort(similarities)[-10:][::-1]
    return [(all_users[i]['dummy_username'], similarities[i]) for i in top10_idx]

# Streamlit UI
st.title("Intern Connector App")

# Info Box
users = load_users_from_blob()
st.info(f"""
- Registered users can directly query. Registered username is needed for querying.
- Please enter unique dummy usernames and not original names (usernames would be stored).
- **Current number of users: {len(users)}**
- **NOTE**: This is a personal project so please use it responsibly. Avoid sharing overly-personal details. While details are collected to generate embeddings, they are promptly deleted and NOT STORED. The embeddings are stored.
- **WORKING**: User data is converted into a meaningful sentence , passed into a sentence-embedder and cosine-similarity is used to search . This is a very naive-way of doing trying to mimic AI-Search+RAG Systems . 
- **SAMPLE QUERY**: [Can be in Natural language] : An intern who likes to play badminton , is at building 1 and is a software engineer , and knows web deployment and likes chocolate milkshake
- Please enter **ONLY UPTO MAX 3** entires in each field separated by commas (Eg : Gardening , Reading , Singing)
- Queries are allowed only if >20 registered users
""")

# Registration
st.header("Register")
with st.form("register_form"):
    username = st.text_input("Username")
    fav_sports = st.text_input("Favorite Sports")
    fav_music_genres = st.text_input("Favorite Music Genres")
    hobbies = st.text_input("Hobbies")
    fav_movies_or_series = st.text_input("Favorite Movies or Series")
    known_languages = st.text_input("Known Languages")
    tech_interests = st.text_input("Technical Interests")
    intern_or_fte = st.selectbox("Enter_0_for_intern_or_Enter_1_for_fte", [1, 0])
    role = st.text_input("Role")
    division_team = st.text_input("Division/Team")
    building_number = st.text_input("Building Number")
    fav_foods = st.text_input("Favourite foods")
    submit = st.form_submit_button("Register")

    if submit:
        if any(user['dummy_username'] == username for user in users):
            st.error("Username already exists.")
        else:
            user_entry = {
                "dummy_username": username,
                "fav_sports": fav_sports,
                "fav_music_genres": fav_music_genres,
                "hobbies": hobbies,
                "fav_movies_or_series": fav_movies_or_series,
                "known_languages": known_languages,
                "tech_interests": tech_interests,
                "intern_or_fte": intern_or_fte,
                "role": role,
                "division_team": division_team,
                "building_number": building_number,
                "fav_foods": fav_foods
            }
            user_entry["embedding"] = get_embedding(build_sentence(user_entry))
            user_entry = {"dummy_username": user_entry["dummy_username"] , "embedding": user_entry["embedding"]}
            add_user_to_blob(user_entry)
            st.success("User registered successfully!")

# Query
st.header("Find Similar Users")
query_username = st.text_input("Enter your registered username:")
query = st.text_input("Enter your query:")

if st.button("Search"):
    if not any(user['dummy_username'] == query_username for user in users):
        st.warning("Username not found. Please register first.")
    elif len(users) < 20:
        st.warning("Querying is only allowed when there are more than 20 users in the database.")
    else:
        results = find_top10_matches(query, users)
        st.write("Top 10 Matches:")
        for r, score in results:
            st.write(f"- {r} (Similarity Score: {score:.2f})")

