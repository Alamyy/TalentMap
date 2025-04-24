import streamlit as st
import pandas as pd
import pickle
import gdown
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

# Existing code for loading player data
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

@st.cache_data
def load_players():
    file_id = "1YLWNW8n4eFQgG77MILXiRkhWJU5a6r41"
    output_path = "players.pkl"
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return pd.read_pickle(output_path)

@st.cache_data
def load_filters():
    url = "https://raw.githubusercontent.com/Alamyy/TalentMap/main/filters.csv"
    return pd.read_csv(url)

players = load_players()
filters = load_filters()

# Function for finding similar players
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

# Player Info Page
def player_info_page():
    selected_player = st.session_state.selected_player

    # Get player data
    player_data = players[players['name'] == selected_player].iloc[0]
    
    # Display Basic Info
    st.subheader(f"Player Info: {player_data['name']}")
    st.write(f"**Full Name**: {player_data['full_name']}")
    st.write(f"**Description**: {player_data['description']}")
    st.write(f"**Height**: {player_data['height_cm']} cm")
    st.write(f"**Weight**: {player_data['weight_kg']} kg")
    st.write(f"**Date of Birth**: {player_data['dob']}")

    # Display Football Stats
    st.subheader("Football Stats")
    st.write(f"**Overall Rating**: {player_data['overall_rating']}")
    st.write(f"**Potential**: {player_data['potential']}")
    st.write(f"**Preferred Foot**: {player_data['preferred_foot']}")
    st.write(f"**Weak Foot**: {player_data['weak_foot']}")
    st.write(f"**Skill Moves**: {player_data['skill_moves']}")
    st.write(f"**Work Rate**: {player_data['work_rate']}")
    st.write(f"**Specialities**: {player_data['specialities']}")

    # Display Club Info
    st.subheader("Club Info")
    st.write(f"**Club**: {player_data['club_name']}")
    st.write(f"**Club League**: {player_data['club_league_name']}")
    st.write(f"**Club Rating**: {player_data['club_rating']}")
    st.write(f"**Position at Club**: {player_data['club_position']}")

    # Display Country Info
    st.subheader("Country Info")
    st.write(f"**Country**: {player_data['country_name']}")
    st.image(player_data['country_flag'], width=50)
    
    # Display Key Attributes
    st.subheader("Key Attributes")
    st.write(f"**Acceleration**: {player_data['acceleration']}")
    st.write(f"**Sprinting Speed**: {player_data['sprint_speed']}")
    st.write(f"**Dribbling**: {player_data['dribbling']}")
    st.write(f"**Finishing**: {player_data['finishing']}")
    st.write(f"**Passing**: {player_data['short_passing']} (Short), {player_data['long_passing']} (Long)")
    st.write(f"**Defensive Awareness**: {player_data['defensive_awareness']}")
    st.write(f"**Strength**: {player_data['strength']}")
    st.write(f"**Stamina**: {player_data['stamina']}")

    # Add a back button to navigate back to the Find Similar Players page
    if st.button("Back to Player Finder"):
        st.session_state.page = "find_similar_players"

# UI Setup
st.title("🎯 Similar Players Finder")

# Page navigation logic
if "page" not in st.session_state:
    st.session_state.page = "find_similar_players"

# Player Finder Page
if st.session_state.page == "find_similar_players":
    player_names = sorted(players['name'].dropna().unique())
    name = st.selectbox("Choose a player", [''] + player_names)
    top_n = st.slider("Number of similar players to show", 1, 20, 10)

    # Advanced Filters Section
    with st.expander("⚙️ Advanced Filters"):
        max_wage = st.slider("Max Wage (€)", 0, int(filters['wage'].max()), 0, step=5000)
        max_value = st.slider("Max Value (€)", 0, int(filters['value'].max()), 0, step=5000)
        max_release_clause = st.slider("Max Release Clause (€)", 0, int(filters['release_clause'].max()), 0, step=5000)
        max_age = st.number_input("Max Age", min_value=0, step=1)
        min_overall_rating = st.number_input("Min Overall Rating", min_value=0, step=1)
        club_name = st.text_input("Club Name")
        club_league_name = st.text_input("Club League Name")
        country_name = st.text_input("Country Name")

    # Find similar players based on selection
    if name:
        st.session_state.selected_player = name
        st.session_state.page = "player_info_page"
        st.write(f"**Searching for similar players to {name}...**")

if st.session_state.page == "player_info_page":
    player_info_page()
