import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


class SpatialClusterAggregator:
    """
    Spatially-aware aggregation of patch-level anomaly scores.

    Input:
        coordinates: (N, 2) array containing patch x,y coordinates
        scores:      (N,) array containing patch anomaly scores

    Output:
        slide-level anomaly score
        cluster-level information
    """

    def __init__(
        self,
        score_threshold=0.5,
        eps=2.0,
        min_samples=2,
        alpha=0.4,   # max anomaly score
        beta=0.3,    # mean anomaly score
        gamma=0.2,   # cluster size
        delta=0.1,   # spatial density
        normalize=True   # per-slide MinMax of the scores (set False to keep raw probabilities)
    ):
        self.normalize = normalize
        self.score_threshold = score_threshold
        self.eps = eps
        self.min_samples = min_samples

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def aggregate(self, coordinates, scores):

        coordinates = np.asarray(coordinates)
        scores = np.asarray(scores)

        assert len(coordinates) == len(scores)

        # ---------------------------------------------------------
        # Step 1: Normalize anomaly scores
        # ---------------------------------------------------------

        if self.normalize:
            # NOTE: this is PER SLIDE, so every slide's top patch becomes 1.0 and the
            # absolute anomaly level is lost. Set normalize=False to keep raw scores.
            scores_norm = MinMaxScaler().fit_transform(
                scores.reshape(-1, 1)
            ).ravel()
        else:
            scores_norm = scores

        # ---------------------------------------------------------
        # Step 2: Select suspicious patches
        # ---------------------------------------------------------

        suspicious_mask = scores_norm >= self.score_threshold

        suspicious_coords = coordinates[suspicious_mask]
        suspicious_scores = scores_norm[suspicious_mask]

        if len(suspicious_coords) == 0:
            return 0.0, []

        # ---------------------------------------------------------
        # Step 3: Spatial clustering
        # ---------------------------------------------------------

        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        ).fit(suspicious_coords)

        labels = clustering.labels_

        # ---------------------------------------------------------
        # Step 4: Extract clusters
        # ---------------------------------------------------------

        clusters = []

        unique_labels = set(labels)

        # DBSCAN label -1 = noise
        unique_labels.discard(-1)

        total_suspicious = len(suspicious_coords)

        for cluster_id in unique_labels:

            mask = labels == cluster_id

            cluster_coords = suspicious_coords[mask]
            cluster_scores = suspicious_scores[mask]

            n_patches = len(cluster_scores)

            # ---------------------------------------------
            # Anomaly intensity
            # ---------------------------------------------

            max_score = np.max(cluster_scores)

            mean_score = np.mean(cluster_scores)

            # ---------------------------------------------
            # Cluster size
            # ---------------------------------------------

            size_score = np.log1p(n_patches)

            # Normalize approximately to [0,1]
            size_score = size_score / np.log1p(total_suspicious)

            # ---------------------------------------------
            # Spatial density
            # ---------------------------------------------

            if n_patches > 1:

                # Bounding box
                x_min = np.min(cluster_coords[:, 0])
                x_max = np.max(cluster_coords[:, 0])

                y_min = np.min(cluster_coords[:, 1])
                y_max = np.max(cluster_coords[:, 1])

                area = max(
                    (x_max - x_min + 1) *
                    (y_max - y_min + 1),
                    1
                )

                density = n_patches / area

                # Convert to bounded score
                density_score = density / (density + 1.0)

            else:
                density_score = 0.0

            # ---------------------------------------------
            # Cluster score
            # ---------------------------------------------

            cluster_score = (
                self.alpha * max_score +
                self.beta * mean_score +
                self.gamma * size_score +
                self.delta * density_score
            )

            clusters.append({
                "cluster_id": int(cluster_id),
                "score": float(cluster_score),
                "max_score": float(max_score),
                "mean_score": float(mean_score),
                "n_patches": int(n_patches),
                "density": float(density_score),
                "coordinates": cluster_coords
            })

        # ---------------------------------------------------------
        # Step 5: Slide-level aggregation
        # ---------------------------------------------------------

        if len(clusters) == 0:

            # Only isolated suspicious patches
            slide_score = float(np.max(suspicious_scores))

        else:

            # The strongest spatially coherent lesion
            slide_score = max(
                cluster["score"]
                for cluster in clusters
            )

            # Compare against isolated high-scoring patches
            isolated_scores = []

            noise_mask = labels == -1

            if np.any(noise_mask):
                isolated_scores = suspicious_scores[noise_mask]

            if len(isolated_scores) > 0:
                slide_score = max(
                    slide_score,
                    float(np.max(isolated_scores))
                )

        return float(slide_score), clusters