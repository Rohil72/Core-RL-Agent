import numpy as np
import pandas as pd

from src.memory.market_memory import MarketMemoryConfig, score_market_memory
from src.memory.retrieval import (
    RetrievalConfig,
    cap_same_ticker_neighbors,
    deduplicate_close_neighbors,
    retrieve_neighbors,
)


def _memory_frame():
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "CCC"],
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-02", "2024-01-01"], utc=True),
            "latent_0": [0.0, 0.1, 0.2, 0.0],
            "latent_1": [0.0, 0.0, 0.0, 0.0],
            "future_max_return_63": [0.20, 0.10, 0.08, 0.99],
            "future_min_return_63": [-0.03, -0.05, -0.02, -0.80],
            "event_upside_before_drawdown_126": [1.0, 0.0, 1.0, 0.0],
        }
    )


def test_temporal_retrieval_excludes_future_neighbors():
    query = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "timestamp": pd.to_datetime(["2023-06-01"], utc=True),
            "latent_0": [0.0],
            "latent_1": [0.0],
        }
    )
    signals, neighbors = score_market_memory(_memory_frame(), query, MarketMemoryConfig(k=4))

    assert signals.loc[0, "retrieval_neighbor_count"] == 3
    assert pd.to_datetime(neighbors["neighbor_timestamp"]).max() < query.loc[0, "timestamp"]
    assert "CCC" not in set(neighbors["neighbor_ticker"])


def test_outcome_availability_filter_handles_microsecond_parquet_timestamps():
    memory = pd.DataFrame(
        {
            "ticker": ["PAST", "FUTURE"],
            "timestamp": pd.to_datetime(["2021-01-01", "2021-01-02"], utc=True),
            "outcome_available_timestamp": pd.Series(
                [pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")],
                dtype="datetime64[us, UTC]",
            ),
        }
    )
    memory_x = np.asarray([[0.0], [0.0]], dtype=np.float32)

    indices, _ = retrieve_neighbors(
        memory,
        memory_x,
        np.asarray([0.0], dtype=np.float32),
        pd.Timestamp("2023-01-01", tz="UTC"),
        "QUERY",
        RetrievalConfig(k=2, require_outcome_availability=True),
    )

    assert indices.tolist() == [0]


def test_same_ticker_cap_and_score_are_deterministic():
    query = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "timestamp": pd.to_datetime(["2023-06-01"], utc=True),
            "latent_0": [0.0],
            "latent_1": [0.0],
        }
    )
    cfg = MarketMemoryConfig(k=3, same_ticker_neighbor_limit=0.34)
    signals_a, neighbors_a = score_market_memory(_memory_frame(), query, cfg)
    signals_b, neighbors_b = score_market_memory(_memory_frame(), query, cfg)

    assert (neighbors_a["neighbor_ticker"] == "AAA").sum() <= 1
    assert np.isclose(signals_a.loc[0, "opportunity_score"], signals_b.loc[0, "opportunity_score"])
    assert neighbors_a["neighbor_ticker"].tolist() == neighbors_b["neighbor_ticker"].tolist()


def test_progressive_partition_matches_full_filtered_sort():
    rng = np.random.default_rng(7)
    count = 500
    memory_x = rng.normal(size=(count, 8)).astype(np.float32)
    tickers = np.asarray([f"T{index % 12}" for index in range(count)])
    memory = pd.DataFrame(
        {
            "ticker": tickers,
            "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(np.arange(count), unit="D"),
            "outcome_available_timestamp": pd.Timestamp("2021-06-01", tz="UTC"),
            "session_index": np.arange(count),
        }
    )
    query = rng.normal(size=8).astype(np.float32)
    cfg = RetrievalConfig(
        k=25,
        causal_horizon_sessions=0,
        minimum_neighbor_separation_sessions=5,
        same_ticker_neighbor_limit=0.20,
        require_outcome_availability=True,
    )
    actual, actual_distances = retrieve_neighbors(
        memory,
        memory_x,
        query,
        pd.Timestamp("2023-01-01", tz="UTC"),
        "T0",
        cfg,
    )

    distances = np.linalg.norm(memory_x - query, axis=1)
    ordered = np.lexsort((np.arange(count), distances))
    expected = deduplicate_close_neighbors(ordered, tickers, np.arange(count), 5)
    expected = cap_same_ticker_neighbors(expected, tickers, "T0", 25, 0.20)

    assert actual.tolist() == expected.tolist()
    assert np.allclose(actual_distances, distances[expected])
