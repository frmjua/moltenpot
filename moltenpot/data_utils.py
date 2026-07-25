"""
data_utils.py — Dataset loading utilities for offline RL training.
==================================================================

Provides PyTorch Dataset classes that stream trajectory sequences directly
from the HDF5 files produced by the data collection pipeline, without
loading the full dataset into memory.

Single-scenario classes
-----------------------
MoltenpotDataset            — (obs, actions) for BC
MoltenpotTransitionDataset  — (obs, actions, rewards, dones) for BCQ/IQL/CQL

Multi-scenario classes
----------------------
MultiScenarioDataset            — same as above, across N scenario files
MultiScenarioTransitionDataset  — same as above, across N scenario files

The multi-scenario variants sample uniformly across scenarios in expectation,
regardless of the number of episodes or focal agents per scenario.  Pass
``dataset.make_sampler()`` as the ``sampler`` argument to DataLoader.

HDF5 schema expected
--------------------
/<scenario_name>/
    ep_0/
        obs       (T, A, 3, 88, 88)  uint8   LZF-compressed
        actions   (T, A)             int32
        rewards   (T, A)             float32
        dones     (T,)               bool
        h_states  (T+1, A, H)        float32
    ep_1/
        ...
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import h5py
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from moltenpot.hub import ensure_dataset, ensure_datasets

logger = logging.getLogger(__name__)


class MoltenpotDataset(Dataset):
    """
    Offline dataset for algorithms that only need (obs, actions).

    Used by Behavior Cloning.

    Each sample is a fixed-length sequence ``(obs, actions)`` for a single
    focal agent extracted from the HDF5 file.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 dataset file.
    scenario : str
        Scenario key inside the HDF5 file (e.g. ``"clean_up_0"``).
    seq_len : int
        Length of each sampled sequence.
    """

    def __init__(self, hdf5_path: str, scenario: str, seq_len: int = 128) -> None:
        if not os.path.isfile(hdf5_path):
            _data_root = os.path.dirname(os.path.dirname(os.path.abspath(hdf5_path)))
            ensure_dataset(_data_root, scenario)

        self.hdf5_path = hdf5_path
        self.scenario  = scenario
        self.seq_len   = seq_len
        self.samples   = []

        with h5py.File(self.hdf5_path, "r") as f:
            if self.scenario not in f:
                raise ValueError(f"Scenario '{scenario}' not found in {hdf5_path}")

            grp = f[self.scenario]
            for ep_name in grp.keys():
                ep = grp[ep_name]
                T = ep["obs"].shape[0]
                A = ep["obs"].shape[1]
                for start_t in range(0, T - self.seq_len + 1, self.seq_len):
                    for a in range(A):
                        self.samples.append((ep_name, a, start_t))

        logger.info(
            "MoltenpotDataset: %d sequences (len=%d) from %s/%s",
            len(self.samples), seq_len, hdf5_path, scenario,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ep_name, agent_idx, start_t = self.samples[idx]
        end_t = start_t + self.seq_len

        with h5py.File(self.hdf5_path, "r") as f:
            grp = f[self.scenario][ep_name]
            obs_uint8 = grp["obs"][start_t:end_t, agent_idx]      # (T, 3, 88, 88)
            actions   = grp["actions"][start_t:end_t, agent_idx]  # (T,)

        obs = torch.from_numpy(obs_uint8).float() / 255.0
        act = torch.from_numpy(actions).long()
        return obs, act


class MoltenpotTransitionDataset(Dataset):
    """
    Offline dataset for algorithms that need (obs, next_obs, actions, rewards, dones).

    Used by BCQ, IQL, and CQL. Samples a window of length T+1 to provide
    both current and next observations for TD-style updates.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 dataset file.
    scenario : str
        Scenario key inside the HDF5 file (e.g. ``"clean_up_0"``).
    seq_len : int
        Length T of each sampled sequence (next obs uses T+1 window).
    """

    def __init__(self, hdf5_path: str, scenario: str, seq_len: int = 128) -> None:
        if not os.path.isfile(hdf5_path):
            _data_root = os.path.dirname(os.path.dirname(os.path.abspath(hdf5_path)))
            ensure_dataset(_data_root, scenario)

        self.hdf5_path = hdf5_path
        self.scenario  = scenario
        self.seq_len   = seq_len
        self.samples   = []

        with h5py.File(self.hdf5_path, "r") as f:
            if self.scenario not in f:
                raise ValueError(f"Scenario '{scenario}' not found in {hdf5_path}")

            grp = f[self.scenario]
            for ep_name in grp.keys():
                ep = grp[ep_name]
                obs_key = "obs" if "obs" in ep else "observation"
                T = ep[obs_key].shape[0]
                A = ep[obs_key].shape[1]
                for start_t in range(0, T - self.seq_len, self.seq_len):
                    for a in range(A):
                        self.samples.append((ep_name, a, start_t, obs_key))

        logger.info(
            "MoltenpotTransitionDataset: %d sequences (len=%d) from %s/%s",
            len(self.samples), seq_len, hdf5_path, scenario,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_name, agent_idx, start_t, obs_key = self.samples[idx]
        end_t = start_t + self.seq_len + 1  # +1 for next_obs

        with h5py.File(self.hdf5_path, "r") as f:
            grp = f[self.scenario][ep_name]
            obs_raw = grp[obs_key][start_t:end_t, agent_idx]       # (T+1, 3, 88, 88)
            actions = grp["actions"][start_t:end_t - 1, agent_idx] # (T,)
            rewards = grp["rewards"][start_t:end_t - 1, agent_idx] # (T,)
            dones   = grp["dones"][start_t:end_t - 1]              # (T,)

        obs = torch.from_numpy(obs_raw).float() / 255.0
        act = torch.from_numpy(actions).long()
        rew = torch.from_numpy(rewards).float()
        don = torch.from_numpy(dones).float()

        return obs, act, rew, don


# Backward-compatible alias
IQLMoltenpotDataset = MoltenpotTransitionDataset


# ===========================================================================
# Multi-scenario datasets
# ===========================================================================

def _build_index(
    data_root: str,
    scenario_names: List[str],
    seq_len: int,
    need_next_obs: bool,
) -> Tuple[list, list]:
    """
    Build a flat sample index across all scenarios.

    Each entry is ``(hdf5_path, scenario_key, ep_name, agent_idx, start_t)``.
    Per-sample weights are set so every scenario has equal total weight,
    giving uniform scenario sampling regardless of episode count or agent count.

    Parameters
    ----------
    need_next_obs : bool
        If True, sequences require T+1 observations (transition datasets).
        The effective episode window is shorter by one step.
    """
    # Batch-check all scenarios and offer to download any that are missing.
    ensure_datasets(data_root, scenario_names)

    samples: list = []
    weights: list = []

    for scenario in scenario_names:
        hdf5_path = os.path.join(data_root, scenario, "dataset.hdf5")

        scenario_samples: list = []
        step = seq_len
        min_len = seq_len + (1 if need_next_obs else 0)

        with h5py.File(hdf5_path, "r") as f:
            if scenario not in f:
                raise ValueError(
                    f"Scenario key '{scenario}' not found inside {hdf5_path}"
                )
            grp = f[scenario]
            for ep_name in grp.keys():
                ep = grp[ep_name]
                T = ep["obs"].shape[0]
                A = ep["obs"].shape[1]
                for start_t in range(0, T - min_len + 1, step):
                    for a in range(A):
                        scenario_samples.append(
                            (hdf5_path, scenario, ep_name, a, start_t)
                        )

        n = len(scenario_samples)
        if n == 0:
            logger.warning("No valid sequences for scenario '%s', skipping.", scenario)
            continue

        # weight[j] = 1/n  →  total weight per scenario = 1.0
        # WeightedRandomSampler normalises across all samples,
        # so equal total weight ⟺ equal sampling probability per scenario.
        samples.extend(scenario_samples)
        weights.extend([1.0 / n] * n)

        logger.info(
            "MultiScenarioDataset: %4d sequences (seq_len=%d) from %s",
            n, seq_len, scenario,
        )

    if not samples:
        raise RuntimeError("No samples found across all provided scenarios.")

    return samples, weights


class MultiScenarioDataset(Dataset):
    """
    Streams ``(obs, actions)`` sequences from multiple HDF5 scenario files.

    Scenarios with more focal agents or more episodes are *not* over-represented.
    Use ``dataset.make_sampler()`` as the ``sampler`` argument to DataLoader
    to enforce uniform scenario sampling.

    Parameters
    ----------
    data_root : str
        Root directory containing one sub-directory per scenario, each
        holding a ``dataset.hdf5`` file.
    scenario_names : list of str
        Scenario keys to load (e.g. ``["clean_up_0", "clean_up_2"]``).
    seq_len : int
        Length of each sampled sequence.
    """

    def __init__(
        self,
        data_root: str,
        scenario_names: List[str],
        seq_len: int = 128,
    ) -> None:
        self.seq_len = seq_len
        self._samples, self._weights = _build_index(
            data_root, scenario_names, seq_len, need_next_obs=False
        )
        logger.info(
            "MultiScenarioDataset ready: %d total sequences across %d scenarios",
            len(self._samples), len(scenario_names),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hdf5_path, scenario, ep_name, agent_idx, start_t = self._samples[idx]
        end_t = start_t + self.seq_len

        with h5py.File(hdf5_path, "r") as f:
            grp = f[scenario][ep_name]
            obs_uint8 = grp["obs"][start_t:end_t, agent_idx]       # (T, 3, 88, 88)
            actions   = grp["actions"][start_t:end_t, agent_idx]   # (T,)

        obs = torch.from_numpy(obs_uint8).float() / 255.0
        act = torch.from_numpy(actions).long()
        aid = torch.tensor(agent_idx, dtype=torch.long)
        return obs, act, aid

    def make_sampler(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler that gives equal weight per scenario."""
        return WeightedRandomSampler(
            self._weights, num_samples=len(self._weights), replacement=True
        )


class MultiScenarioTransitionDataset(Dataset):
    """
    Streams ``(obs, actions, rewards, dones)`` sequences from multiple HDF5
    scenario files, with T+1 observations for TD-style offline RL algorithms.

    Use ``dataset.make_sampler()`` as the ``sampler`` argument to DataLoader.

    Parameters
    ----------
    data_root : str
        Root directory containing one sub-directory per scenario.
    scenario_names : list of str
        Scenario keys to load.
    seq_len : int
        Length T of each sampled sequence (window reads T+1 observations).
    """

    def __init__(
        self,
        data_root: str,
        scenario_names: List[str],
        seq_len: int = 128,
    ) -> None:
        self.seq_len = seq_len
        self._samples, self._weights = _build_index(
            data_root, scenario_names, seq_len, need_next_obs=True
        )
        logger.info(
            "MultiScenarioTransitionDataset ready: %d total sequences across %d scenarios",
            len(self._samples), len(scenario_names),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hdf5_path, scenario, ep_name, agent_idx, start_t = self._samples[idx]
        end_t = start_t + self.seq_len + 1  # +1 for next_obs

        with h5py.File(hdf5_path, "r") as f:
            grp = f[scenario][ep_name]
            obs_raw = grp["obs"][start_t:end_t, agent_idx]             # (T+1, 3, 88, 88)
            actions  = grp["actions"][start_t:end_t - 1, agent_idx]    # (T,)
            rewards  = grp["rewards"][start_t:end_t - 1, agent_idx]    # (T,)
            dones    = grp["dones"][start_t:end_t - 1]                  # (T,)

        obs = torch.from_numpy(obs_raw).float() / 255.0
        act = torch.from_numpy(actions).long()
        rew = torch.from_numpy(rewards).float()
        don = torch.from_numpy(dones).float()
        aid = torch.tensor(agent_idx, dtype=torch.long)
        return obs, act, rew, don, aid

    def make_sampler(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler that gives equal weight per scenario."""
        return WeightedRandomSampler(
            self._weights, num_samples=len(self._weights), replacement=True
        )
