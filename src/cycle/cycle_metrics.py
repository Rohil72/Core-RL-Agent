import numpy as np


def latent_velocity(Z):
    """
    Z: (T, D)
    returns v: (T-1, D)
    """
    return Z[1:] - Z[:-1]


def cosine_similarity(a, b, eps=1e-8):
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / (den + eps)


def compute_cycle_score(
    Z,
    window=20,
    coherence_threshold=0.3,
):
    """
    Z: (T, D) latent trajectory
    returns cycle_score: (T,)
    """

    T = len(Z)
    scores = np.zeros(T)

    if T < window + 2:
        return scores

    v = latent_velocity(Z)              # (T-1, D)
    v_norm = np.linalg.norm(v, axis=1)

    # historical baseline for normalization
    baseline = np.median(v_norm[:window]) + 1e-8

    for t in range(window + 1, T - 1):
        # windowed velocities
        v_win = v[t - window : t]
        v_prev = v[t - window - 1 : t - 1]

        # directional coherence
        cos = cosine_similarity(v_win, v_prev)
        coherence = np.mean(cos)

        # persistence
        persistence = np.mean(cos > coherence_threshold)

        # energy (clipped, normalized)
        energy = np.clip(v_norm[t] / baseline, 0.5, 3.0)

        scores[t] = coherence * persistence * energy

    return scores
