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

# Load datasets
players = load_players()
filters = load_filters()

def find_similar_players(input_name, top_n=10, max_wage=None, max_age=None, max_value=None, 
                          max_release_clause=None, club_name=None, club_league_name=None, 
                          country_name=None, min_overall_rating=None):

    matches = players[players['name'] == input_name]
    if matches.empty:
        return "❌ No matching player found.", []

    results = []

    for _, match_row in matches.iterrows():
        player_name = match_row['name']
        player_id = match_row['player_id']

        position_cols = [col for col in players.columns if col in position_data]
        player_position = next((pos for pos in position_cols if match_row.get(pos, 0) == 1), None)

        if not player_position or player_position not in position_data:
            return f"⚠️ No data for position: {player_position}", []

        dataset = pd.read_pickle(position_data[player_position]['dataset_path'])
        with open(position_data[player_position]['features_path'], "rb") as f:
            features = pickle.load(f)

        if player_id not in dataset['player_id'].values:
            return "❌ Player not found in position dataset.", []

        df = dataset[['name', 'player_id'] + features].dropna()
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)

        idx = df[df['player_id'] == player_id].index[0]
        input_vector = X_pca[idx]

        eligible_players = []

        for i, candidate_row in df.iterrows():
            sim_id = candidate_row['player_id']
            if sim_id == player_id:
                continue

            valid = True

            def check(column, value, op):
                val = filters.loc[filters['player_id'] == sim_id, column]
                return not val.empty and op(val.iloc[0], value)

            if max_wage and not check('wage', max_wage, lambda a, b: a <= b): valid = False
            if max_value and not check('value', max_value, lambda a, b: a <= b): valid = False
            if max_release_clause and not check('release_clause', max_release_clause, lambda a, b: a <= b): valid = False
            if max_age and not check('age', max_age, lambda a, b: a <= b): valid = False
            if club_name and not check('club_name', club_name, lambda a, b: a == b): valid = False
            if club_league_name and not check('club_league_name', club_league_name, lambda a, b: a == b): valid = False
            if country_name and not check('country_name', country_name, lambda a, b: a == b): valid = False
            if min_overall_rating and not check('overall_rating', min_overall_rating, lambda a, b: a >= b): valid = False

            if valid:
                pca_idx = df.index.get_loc(i)
                candidate_vector = X_pca[pca_idx]
                score = 1 - cosine_distances([input_vector], [candidate_vector])[0][0]
                eligible_players.append((candidate_row['name'], score))

        eligible_players.sort(key=lambda x: x[1], reverse=True)
        results.extend(eligible_players[:top_n])

    return f"🔍 Similar players to {input_name}:", results[:top_n]

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
        st.table(pd.DataFrame(results, columns=["Player Name", "Similarity Score"]))
