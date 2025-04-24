import streamlit as st
import pandas as pd
import pickle
import gdown
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

position_data = {
    'CAM': {'dataset_path': 'attack_mid.pkl', 'features_path': 'attack_mid_features.pkl'},
    'LW': {'dataset_path': 'wingers.pkl', 'features_path': 'wingers_features.pkl'},
    'RW': {'dataset_path': 'wingers.pkl', 'features_path': 'wingers_features.pkl'},
    'ST': {'dataset_path': 'strikers.pkl', 'features_path': 'strikers_features.pkl'},
    'CDM': {'dataset_path': 'defensive_mids.pkl', 'features_path': 'defensive_mids_features.pkl'},
    'CM': {'dataset_path': 'defensive_mids.pkl', 'features_path': 'defensive_mids_features.pkl'},
    'CB': {'dataset_path': 'center_backs.pkl', 'features_path': 'center_backs_features.pkl'},
    'LB': {'dataset_path': 'full_backs.pkl', 'features_path': 'full_backs_features.pkl'},
    'RB': {'dataset_path': 'full_backs.pkl', 'features_path': 'full_backs_features.pkl'}
}

st.set_page_config(page_title="TalentMap", page_icon="⚽")

@st.cache_data
def load_players():
    url = "https://raw.githubusercontent.com/Alamyy/TalentMap/refs/heads/main/players_adjusted.csv"
    return pd.read_csv(url)

@st.cache_data
def load_filters():
    url = "https://raw.githubusercontent.com/Alamyy/TalentMap/main/filters.csv"
    return pd.read_csv(url)
    
@st.cache_data
def load_moreinfo():
    url = "https://raw.githubusercontent.com/Alamyy/TalentMap/refs/heads/main/player-data-full.csv"
    return pd.read_csv(url)

# Load datasets
players = load_players()
filters = load_filters()
moreinfo = load_moreinfo()

# Merge the players DataFrame with the moreinfo DataFrame based on the 'name' column
merged_players = pd.merge(players, moreinfo, on='name', how='left')
def find_similar_players(name, top_n, players_data, merged_players):
    # Ensure the player name is valid in the dataset
    player_info = players_data[players_data['name'] == name]
    
    if player_info.empty:
        return f"Player '{name}' not found in the dataset.", []

    # Retrieve the player's ID and other relevant details
    player_id = player_info.iloc[0]['player_id']
    
    # Filter out players that have valid similarity scores (we assume this is already pre-processed)
    similar_players = merged_players[merged_players['player_id_x'] != player_id]
    
    # Calculate similarity scores (for simplicity, assuming this is already done)
    similar_players['similarity_score'] = calculate_similarity(player_id, similar_players)  # Add your similarity calculation logic here
    
    # Sort players by similarity score and select top_n
    similar_players = similar_players.sort_values(by='similarity_score', ascending=False).head(top_n)
    
    results = []
    for idx, player in similar_players.iterrows():
        sim_id = player['player_id_y']  # Assuming player_id_y is the correct player ID in merged_players
        similar_player_info = merged_players[merged_players['player_id_y'] == sim_id]
        
        if not similar_player_info.empty:
            # Try to extract the club name, if it exists
            club_name = similar_player_info['club_name'].iloc[0] if 'club_name' in similar_player_info.columns else "Unknown Club"
            
            # Ensure there's no missing data in the player information
            player_data = {
                'name': similar_player_info['name'].iloc[0],
                'similarity_score': player['similarity_score'],
                'club': club_name,
                'value': similar_player_info['value'].iloc[0] if 'value' in similar_player_info.columns else "Unknown Value"
            }
            results.append(player_data)
    
    # If no similar players found, return an appropriate message
    if not results:
        return f"No similar players found for '{name}'.", []
    
    return f"Top {top_n} similar players to '{name}':", results



# UI
st.title("🎯 Similar Players Finder")

player_names = sorted(players['name'].dropna().unique())
name = st.selectbox("Choose a player", [''] + player_names)

top_n = st.slider("Number of similar players to show", 1, 20, 10)

# Expanding arrow for filters
with st.expander("🔧 Advanced Filters"):
    max_wage = st.slider("Max Wage (€)", 0, int(filters['wage'].max()), 0, step=5000)
    max_value = st.slider("Max Value (€)", 0, int(filters['value'].max()), 0, step=5000)
    max_release_clause = st.slider("Max Release Clause (€)", 0, int(filters['release_clause'].max()), 0, step=5000)
    max_age = st.number_input("Max Age", min_value=0, step=1)
    min_overall_rating = st.number_input("Min Overall Rating", min_value=0, step=1)

    club_name = st.selectbox("Club Name", [''] + sorted(filters['club_name'].dropna().unique().tolist()))
    club_league_name = st.selectbox("Club League Name", [''] + sorted(filters['club_league_name'].dropna().unique().tolist()))
    country_name = st.selectbox("Country Name", [''] + sorted(filters['country_name'].dropna().unique().tolist()))

# Search button
if st.button("Find Similar Players") and name:
    msg, results = find_similar_players(name, top_n,
                                        max_wage or None,
                                        max_age or None,
                                        max_value or None,
                                        max_release_clause or None,
                                        club_name or None,
                                        club_league_name or None,
                                        country_name or None,
                                        min_overall_rating or None)
    st.write(msg)
    if results:
        st.table(pd.DataFrame(results, columns=["Player Name", "Similarity Score", "Club", "Value (€)"]))
