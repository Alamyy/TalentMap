import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

def load_position_data():
    return {
        'CAM': {'dataset_path': 'data/attack_mid.pkl', 'features_path': 'data/attack_mid_features.pkl'},
        'LW': {'dataset_path': 'data/wingers.pkl', 'features_path': 'data/wingers_features.pkl'},
        'RW': {'dataset_path': 'data/wingers.pkl', 'features_path': 'data/wingers_features.pkl'},
        'ST': {'dataset_path': 'data/strikers.pkl', 'features_path': 'data/strikers_features.pkl'},
        'CDM': {'dataset_path': 'data/defensive_mids.pkl', 'features_path': 'data/defensive_mids_features.pkl'},
        'CM': {'dataset_path': 'data/defensive_mids.pkl', 'features_path': 'data/defensive_mids_features.pkl'},
        'CB': {'dataset_path': 'data/center_backs.pkl', 'features_path': 'data/center_backs_features.pkl'},
        'LB': {'dataset_path': 'data/full_backs.pkl', 'features_path': 'data/full_backs_features.pkl'},
        'RB': {'dataset_path': 'data/full_backs.pkl', 'features_path': 'data/full_backs_features.pkl'}
    }

def find_similar_players(input_name, players, filters, position_data, top_n=10, **kwargs):
    matches = players[players['name'].str.contains(input_name, case=False, na=False)]
    if matches.empty:
        return "❌ No matching player found.", []

    results = []

    for _, row in matches.iterrows():
        player_name = row['name']
        player_id = row['player_id']
        
        position_cols = [col for col in players.columns if col in position_data]
        player_position = next((pos for pos in position_cols if row.get(pos) == 1), None)

        if not player_position or player_position not in position_data:
            continue

        # Load dataset and features
        dataset = pd.read_pickle(position_data[player_position]['dataset_path'])
        with open(position_data[player_position]['features_path'], "rb") as f:
            features = pickle.load(f)

        if player_id not in dataset['player_id'].values:
            continue

        df = dataset[['name', 'player_id'] + features].dropna()
        names = df['name']
        X = df[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)

        idx_match = df[df['player_id'] == player_id].index
        if idx_match.empty:
            continue
        idx = idx_match[0]
        input_vector = X_pca[idx]

        eligible_players = []

        for i, row in df.iterrows():
            sim_id = row['player_id']
            if sim_id == player_id:
                continue

            filter_conditions = True

            for condition, value in kwargs.items():
                if value is not None:
                    col_value = filters.loc[filters['player_id'] == sim_id, condition].values
                    if col_value.size > 0:
                        if "min_" in condition and col_value[0] < value:
                            filter_conditions = False
                        elif "max_" in condition and col_value[0] > value:
                            filter_conditions = False
                        elif condition in ["club_name", "club_league_name", "country_name"] and col_value[0] != value:
                            filter_conditions = False

            if filter_conditions:
                pca_idx = df.index.get_loc(i)
                candidate_vector = X_pca[pca_idx]
                similarity_score = 1 - cosine_distances([input_vector], [candidate_vector])[0][0]
                eligible_players.append((row['name'], similarity_score))

        eligible_players.sort(key=lambda x: x[1], reverse=True)
        results = eligible_players[:top_n]
        break

    if results:
        return f"✅ Similar players to: {player_name}", results
    return "⚠️ No players meet the filter criteria.", []
