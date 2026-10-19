from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLPBlock(nn.Module):
    """
    Residual MLP block with LayerNorm:
        x -> Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> (+ x) -> ReLU
    Using LayerNorm ensures single-leaf MCTS inference (B=1) and minibatch training (B=64)
    compute identical feature normalizations.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        return F.relu(out + residual)


class SortingProgramNet(nn.Module):
    """
    Split-Architecture Dual-Headed Network for Single-Player Sorting Program Synthesis:
    - Input `obs` has shape `(B, 2^N + 1)`:
        * `failing_cases = obs[:, :-1]` of shape `(B, 2^N)` (time-independent Markov state of test cases)
        * `step_frac     = obs[:, -1:]` of shape `(B, 1)`   (`steps_taken / max_steps` in [0, 1])
    - Shared ResMLP Trunk & Policy Head operate purely on `failing_cases` so the policy prior
      P(a | failing_cases) is 100% time-invariant across depths and transpositions.
    - Value Head concatenates the trunk features with `step_frac` so V(s, t) is time-aware
      with respect to the timeout horizon, and uses `-F.softplus(v)` to guarantee V(s, t) < 0.
    """

    def __init__(
        self,
        num_test_cases: int = 16,
        num_actions: int = 6,
        hidden_dim: int = 128,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.num_test_cases = num_test_cases
        self.num_actions = num_actions

        # Stem + Residual MLP Trunk (operates purely on the 2^N failing test cases)
        self.stem = nn.Sequential(
            nn.Linear(num_test_cases, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.trunk = nn.ModuleList(
            [ResMLPBlock(hidden_dim) for _ in range(num_blocks)]
        )

        # Policy Head (Time-Invariant: depends only on trunk features)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

        # Value Head (Time-Aware: concatenates trunk features [hidden_dim] + step_frac [1])
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        failing_cases = obs[:, :-1]  # (B, 2^N)
        step_frac = obs[:, -1:]      # (B, 1)

        h = self.stem(failing_cases)
        for block in self.trunk:
            h = block(h)

        logits = self.policy_head(h)  # (B, num_actions)

        v_in = torch.cat([h, step_frac], dim=-1)  # (B, hidden_dim + 1)
        raw_v = self.value_head(v_in)             # (B, 1)
        value = -F.softplus(raw_v)                # Strictly non-positive (-inf, 0)

        return logits, value
