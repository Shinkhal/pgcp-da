
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================
# PHASE 1: DISCOVERY & FEATURE SELECTION
# ==========================================

def phase_1_discovery_and_setup():

    print("\n[PHASE 1] Discovery & Feature Selection")
    print("-" * 65)

    df = pd.read_csv("cricket_player_stats.csv")

    print("\nDataset Shape:", df.shape)
    print(df.head())
    print("\nDataset Info:")
    df.info()

    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())

    # Retain original dataframe for profiling
    df_original = df.copy()

    # Remove identifiers
    X = df.drop(columns=["player_id", "player_name"])

    # Select numerical performance features
    X = X.select_dtypes(include=np.number)

    # Handle missing values
    X = X.fillna(X.median())

    print("\nSelected Features:", X.columns.tolist())
    print("Feature Shape:", X.shape)

    return X, df_original


# ==========================================
# PHASE 2: FEATURE SCALING
# ==========================================

def phase_2_feature_scaling(X):

    print("\n[PHASE 2] Unsupervised Feature Scaling")
    print("-" * 65)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("Scaled Feature Shape:", X_scaled.shape)

    return X_scaled, scaler


# ==========================================
# PHASE 3: K-MEANS MODEL INDUCTION
# ==========================================

def phase_3_kmeans_model_induction(X_scaled, df_original):

    print("\n[PHASE 3] K-Means Model Induction")
    print("-" * 65)

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    # Fit and generate labels
    labels = model.fit_predict(X_scaled)

    # Extract centroids
    centroids = model.cluster_centers_

    # Assign clusters
    df_clustered = df_original.copy()
    df_clustered["cluster_id"] = labels

    print("\nClustered Dataset Shape:", df_clustered.shape)
    print("Centroid Shape:", centroids.shape)

    print("\nCluster Counts:")
    print(df_clustered["cluster_id"].value_counts())

    return model, labels, df_clustered


# ==========================================
# PHASE 4: EVALUATION & ROLE PROFILING
# ==========================================

def phase_4_cluster_evaluation_and_profiling(
    X_scaled,
    labels,
    df_clustered,
    model,
    feature_names
):

    print("\n[PHASE 4] Cluster Evaluation & Role Profiling")
    print("-" * 65)

    # Silhouette Score
    score = silhouette_score(X_scaled, labels)

    print(f"\nSilhouette Score: {score:.4f}")

    # Cluster profiles using original statistics
    cluster_profiles = (
        df_clustered
        .groupby("cluster_id")[feature_names]
        .mean()
    )

    print("\nCluster Profiles:")
    print(cluster_profiles)

    # ======================================
    # AUTOMATIC ROLE IDENTIFICATION
    # ======================================

    centroids = model.cluster_centers_

    # Feature positions
    bat_indices = [
        feature_names.index(col)
        for col in ["runs_scored", "batting_average"]
        if col in feature_names
    ]

    bowl_indices = [
        feature_names.index(col)
        for col in ["wickets_taken", "bowling_economy"]
        if col in feature_names
    ]

    # Calculate batting and bowling strength
    bat_scores = {}
    bowl_scores = {}

    for cluster_id in range(3):

        bat_scores[cluster_id] = (
            np.mean(centroids[cluster_id, bat_indices])
            if bat_indices else 0
        )

        bowling_values = []

        for col in ["wickets_taken", "bowling_economy"]:

            if col in feature_names:

                idx = feature_names.index(col)

                value = centroids[cluster_id, idx]

                # More wickets = better
                # Lower economy = better
                if col == "bowling_economy":
                    value = -value

                bowling_values.append(value)

        bowl_scores[cluster_id] = (
            np.mean(bowling_values)
            if bowling_values else 0
        )

    # All-rounder: strongest balance of batting and bowling
    all_rounder_cluster = max(
        range(3),
        key=lambda c: min(
            bat_scores[c],
            bowl_scores[c]
        )
    )

    remaining_clusters = [
        c for c in range(3)
        if c != all_rounder_cluster
    ]

    # Batsman: stronger batting performance
    batsman_cluster = max(
        remaining_clusters,
        key=lambda c: bat_scores[c]
    )

    # Remaining cluster becomes bowler
    bowler_cluster = next(
        c for c in remaining_clusters
        if c != batsman_cluster
    )

    role_mapping = {
        batsman_cluster: "Batsman",
        bowler_cluster: "Bowler",
        all_rounder_cluster: "All-Rounder"
    }

    print("\nDetected Cluster Roles:")
    print(role_mapping)

    # Assign role labels
    df_clustered["role"] = df_clustered["cluster_id"].map(
        role_mapping
    )

    print("\nFinal Player Segmentation:")
    print(
        df_clustered[
            ["player_name", "cluster_id", "role"]
        ].head(15)
    )

    return score, cluster_profiles, df_clustered, role_mapping


# ==========================================
# PHASE 5: JOBLIB MODEL PERSISTENCE
# ==========================================

def phase_5_joblib_model_persistence(
    model,
    scaler,
    feature_names,
    role_mapping
):

    print("\n[PHASE 5] Model Persistence")
    print("-" * 65)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "role_mapping": role_mapping
    }

    # Save
    joblib.dump(artifacts, "cricket_model.joblib")

    print("\nArtifacts saved successfully!")

    # Load
    loaded = joblib.load("cricket_model.joblib")

    loaded_model = loaded["model"]
    loaded_scaler = loaded["scaler"]
    loaded_features = loaded["feature_names"]
    loaded_mapping = loaded["role_mapping"]

    print("Artifacts loaded successfully!")

    # New player sample
    sample = pd.DataFrame(
        [[5000, 45.0, 150, 3.5, 20]],
        columns=loaded_features
    )

    # Scale new data
    sample_scaled = loaded_scaler.transform(sample)

    # Predict cluster
    predicted_cluster = loaded_model.predict(sample_scaled)[0]

    # Identify role
    predicted_role = loaded_mapping[predicted_cluster]

    print("\nNew Player Prediction:")
    print("Cluster ID:", predicted_cluster)
    print("Predicted Role:", predicted_role)


# ==========================================
# FACADE FUNCTION
# ==========================================

def run_cricket_clustering_pipeline():

    print("=" * 65)
    print("CRICKET PLAYER ROLE CLUSTERING LIFECYCLE")
    print("=" * 65)

    # Phase 1
    X, df_original = phase_1_discovery_and_setup()

    # Phase 2
    X_scaled, scaler = phase_2_feature_scaling(X)

    # Phase 3
    model, labels, df_clustered = (
        phase_3_kmeans_model_induction(
            X_scaled,
            df_original
        )
    )

    # Phase 4
    score, profiles, df_clustered, role_mapping = (
        phase_4_cluster_evaluation_and_profiling(
            X_scaled,
            labels,
            df_clustered,
            model,
            X.columns.tolist()
        )
    )

    # Phase 5
    phase_5_joblib_model_persistence(
        model,
        scaler,
        X.columns.tolist(),
        role_mapping
    )

    print("\nPipeline Completed Successfully!")


# ==========================================
# MAIN GUARD
# ==========================================

if __name__ == "__main__":
    run_cricket_clustering_pipeline()
