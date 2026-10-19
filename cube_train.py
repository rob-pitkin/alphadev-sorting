import argparse
from collections import deque
from dataclasses import dataclass
import os
from typing import Deque, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from cube_env import PocketCubeEnv
from mcts import SinglePlayerMCTS
from model import SortingProgramNet


@dataclass
class CubeSample:
    obs: np.ndarray
    policy_target: np.ndarray
    value_target: float


class PocketCubeTrainer:
    """Trains Single-Player AlphaZero on the 2x2x2 Pocket Rubik's Cube.

    Supports two training regimes to demonstrate the necessity of Backward Curricula:
      1. `mode="naive"`: Every episode starts at fixed deep scramble `k = max_scramble_depth`.
      2. `mode="curriculum"`: Mastery-gated Reverse-Scramble Backward Curriculum
         (`K = 1 -> max_scramble_depth`) with mixed inner-ring sampling (60% frontier `K`,
         40% uniform `{1..K-1}`).
    """

    def __init__(
        self,
        mode: str = "curriculum",
        max_scramble_depth: int = 8,
        max_steps: int = 12,
        num_simulations: int = 45,
        c_puct: float = 1.5,
        lr: float = 2e-3,
        batch_size: int = 64,
        buffer_size: int = 15000,
        seed: int = 42,
    ):
        assert mode in ("curriculum", "naive")
        self.mode = mode
        self.max_scramble_depth = max_scramble_depth
        self.max_steps = max_steps
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.env_template = PocketCubeEnv(max_steps=max_steps)
        self.model = SortingProgramNet(
            num_test_cases=144,  # 24 stickers * 6 one-hot colors
            num_actions=self.env_template.num_actions,
            hidden_dim=128,
            num_blocks=3,
        )
        # Pessimistic value initialization for sparse-goal search:
        # Unseen states default to -softplus(+max_steps) ≈ -max_steps until the
        # backward curriculum wave proves they can reach the solved cube in k steps.
        with torch.no_grad():
            self.model.value_head[-1].weight.mul_(0.1)
            self.model.value_head[-1].bias.fill_(float(max_steps))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.mcts = SinglePlayerMCTS(
            model=self.model,
            c_puct=c_puct,
            discount=1.0,
            dirichlet_alpha=0.4,
            dirichlet_eps=0.20,
        )
        self.replay_buffer: Deque[CubeSample] = deque(maxlen=buffer_size)
        self.current_frontier_k: int = (
            1 if mode == "curriculum" else max_scramble_depth
        )

    def sample_scramble_depth(self) -> int:
        if self.mode == "naive":
            return self.max_scramble_depth
        if self.current_frontier_k == 1:
            return 1
        # Mixed sampling: 60% at active frontier K, 40% uniform across inner rings {1..K-1}
        if self.rng.random() < 0.60:
            return self.current_frontier_k
        return int(self.rng.integers(1, self.current_frontier_k))

    def execute_episode(
        self, scramble_depth: int, explore: bool = True
    ) -> Tuple[List[CubeSample], bool, int]:
        """Plays one episode starting from a `scramble_depth` scramble."""
        env = PocketCubeEnv(max_steps=self.max_steps)
        env.reset(scramble_depth=scramble_depth, rng=self.rng)

        trajectory_obs: List[np.ndarray] = []
        trajectory_pi: List[np.ndarray] = []
        trajectory_rewards: List[float] = []

        step_cap = min(self.max_steps, scramble_depth + 2) if (explore and self.mode == "curriculum") else self.max_steps
        while not env.is_terminal() and env.steps_taken < step_cap:
            obs = env.get_observation()
            pi = self.mcts.search(
                root_env=env,
                num_simulations=self.num_simulations,
                temperature=1.0,
                add_noise=explore,
            )
            action = int(np.argmax(pi))

            trajectory_obs.append(obs)
            trajectory_pi.append(pi)
            _, reward, _ = env.step(action)
            trajectory_rewards.append(reward)

        solved = env.is_solved()
        if not solved and trajectory_rewards:
            # Ensure early-capped curriculum timeout carries the standard timeout penalty
            if not env.is_terminal():
                trajectory_rewards[-1] -= float(self.max_steps)

        T = len(trajectory_rewards)
        returns = np.zeros(T, dtype=np.float32)
        running_g = 0.0
        for t in reversed(range(T)):
            running_g += trajectory_rewards[t]
            returns[t] = running_g

        samples = [
            CubeSample(
                obs=trajectory_obs[t],
                policy_target=trajectory_pi[t],
                value_target=float(returns[t]),
            )
            for t in range(T)
        ]
        if not solved and self.mode == "curriculum":
            # Rejection sampling (Expert Iteration): do not train the network to
            # imitate failed trajectories or mislabel solvable depth-K states as timeouts
            samples = []
        return samples, solved, env.steps_taken

    def train_step(self) -> Tuple[float, float]:
        if len(self.replay_buffer) < 16:
            return 0.0, 0.0

        actual_batch = min(self.batch_size, len(self.replay_buffer))
        indices = self.rng.choice(
            len(self.replay_buffer), size=actual_batch, replace=False
        )
        batch = [self.replay_buffer[i] for i in indices]

        obs_t = torch.from_numpy(np.stack([s.obs for s in batch])).float()
        pi_t = torch.from_numpy(np.stack([s.policy_target for s in batch])).float()
        v_t = torch.tensor(
            [s.value_target for s in batch], dtype=torch.float32
        ).unsqueeze(-1)

        self.model.train()
        logits, v_pred = self.model(obs_t)
        log_probs = F.log_softmax(logits, dim=-1)

        policy_loss = -torch.sum(pi_t * log_probs, dim=-1).mean()
        value_loss = F.smooth_l1_loss(v_pred, v_t)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(policy_loss.item()), float(value_loss.item())

    def evaluate_depth(
        self, depth: int, num_episodes: int = 12, eval_sims: int = 50
    ) -> Tuple[float, float]:
        """Evaluates greedy MCTS solve rate and average solve length at `depth`."""
        solved_count = 0
        solved_lengths: List[int] = []
        old_sims = self.num_simulations
        self.num_simulations = eval_sims
        for _ in range(num_episodes):
            _, solved, steps = self.execute_episode(scramble_depth=depth, explore=False)
            if solved:
                solved_count += 1
                solved_lengths.append(steps)
        self.num_simulations = old_sims
        solve_rate = solved_count / float(num_episodes)
        avg_len = float(np.mean(solved_lengths)) if solved_lengths else float(self.max_steps)
        return solve_rate, avg_len

    def run_training(
        self,
        num_iterations: int = 20,
        episodes_per_iter: int = 24,
        train_steps_per_iter: int = 50,
        save_path: str = "cube_best.pt",
    ) -> None:
        print(
            f"=== Training 2x2x2 Pocket Cube ({self.mode.upper()} mode) | "
            f"Max Scramble K={self.max_scramble_depth} | Max Steps={self.max_steps} ==="
        )
        for it in range(1, num_iterations + 1):
            frontier_solved = 0
            frontier_total = 0
            all_solved = 0

            for _ in range(episodes_per_iter):
                k = self.sample_scramble_depth()
                samples, solved, _ = self.execute_episode(scramble_depth=k, explore=True)
                self.replay_buffer.extend(samples)
                if solved:
                    all_solved += 1
                if k == self.current_frontier_k:
                    frontier_total += 1
                    if solved:
                        frontier_solved += 1

            p_losses, v_losses = [], []
            for _ in range(train_steps_per_iter):
                pl, vl = self.train_step()
                p_losses.append(pl)
                v_losses.append(vl)

            frontier_rate = frontier_solved / max(1, frontier_total)
            overall_rate = all_solved / float(episodes_per_iter)

            status_note = ""
            if (
                self.mode == "curriculum"
                and frontier_rate >= 0.80
                and self.current_frontier_k < self.max_scramble_depth
            ):
                self.current_frontier_k += 1
                status_note = f"  --> [FRONTIER ADVANCED TO K={self.current_frontier_k}!]"

            print(
                f"Iter {it:2d}/{num_iterations} | Frontier K={self.current_frontier_k:2d} | "
                f"Frontier Solve: {frontier_rate*100:5.1f}% | Overall Solve: {overall_rate*100:5.1f}% | "
                f"Loss (P/V): {np.mean(p_losses):.3f}/{np.mean(v_losses):.3f}{status_note}"
            )

        if not os.path.isabs(save_path):
            save_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), save_path
            )
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}\n")


def run_comparison():
    max_depth = 7
    max_steps = 11
    num_iters = 16
    eps_per_iter = 24

    # 1. Train Naive Forward Self-Play (always starts at k = 7)
    naive_trainer = PocketCubeTrainer(
        mode="naive",
        max_scramble_depth=max_depth,
        max_steps=max_steps,
        num_simulations=40,
        seed=42,
    )
    naive_trainer.run_training(
        num_iterations=num_iters,
        episodes_per_iter=eps_per_iter,
        train_steps_per_iter=45,
        save_path="cube_naive.pt",
    )

    # 2. Train Mastery-Gated Reverse-Scramble Backward Curriculum (K = 1 -> 7)
    curr_trainer = PocketCubeTrainer(
        mode="curriculum",
        max_scramble_depth=max_depth,
        max_steps=max_steps,
        num_simulations=40,
        seed=42,
    )
    curr_trainer.run_training(
        num_iterations=num_iters,
        episodes_per_iter=eps_per_iter,
        train_steps_per_iter=45,
        save_path="cube_curriculum.pt",
    )

    # 3. Head-to-Head Benchmark across Scramble Depths k = 1..7
    print("=== HEAD-TO-HEAD BENCHMARK: NAIVE FORWARD vs. BACKWARD CURRICULUM ===")
    print(f"{'Scramble Depth k':>16} | {'Naive Solve %':>15} | {'Curriculum Solve %':>20} | {'Curr Avg Moves':>15}")
    print("-" * 76)
    for k in range(1, max_depth + 1):
        n_rate, _ = naive_trainer.evaluate_depth(k, num_episodes=15, eval_sims=50)
        c_rate, c_len = curr_trainer.evaluate_depth(k, num_episodes=15, eval_sims=50)
        print(
            f"{k:16d} | {n_rate*100:14.1f}% | {c_rate*100:19.1f}% | {c_len:15.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Run head-to-head Naive vs Curriculum benchmark")
    parser.add_argument("--mode", type=str, default="curriculum", choices=["curriculum", "naive"])
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=16)
    args = parser.parse_args()

    if args.compare:
        run_comparison()
    else:
        trainer = PocketCubeTrainer(
            mode=args.mode,
            max_scramble_depth=args.max_depth,
            max_steps=args.max_depth + 4,
        )
        trainer.run_training(
            num_iterations=args.iterations,
            save_path=f"cube_{args.mode}.pt",
        )
