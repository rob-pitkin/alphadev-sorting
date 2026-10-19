import argparse
import random
from collections import deque
from typing import Deque, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from env import SortingProgramEnv
from mcts import SinglePlayerMCTS
from model import SortingProgramNet

# Known theoretical minimum swap_if_greater statements vs. Bubble Sort N*(N-1)//2
OPTIMAL_LENGTHS = {3: 3, 4: 5, 5: 9, 6: 12}


class SinglePlayerTrainer:
    def __init__(
        self,
        array_len: int = 4,
        max_steps: Optional[int] = None,
        num_iterations: int = 8,
        episodes_per_iter: int = 25,
        num_simulations: int = 50,
        batch_size: int = 64,
        epochs_per_iter: int = 6,
        lr: float = 1e-3,
        discount: float = 1.0,
        buffer_maxlen: int = 10_000,
        device: str = "cpu",
    ):
        self.array_len = array_len
        self.bubble_sort_len = array_len * (array_len - 1) // 2
        self.optimal_len = OPTIMAL_LENGTHS.get(array_len, self.bubble_sort_len)
        self.max_steps = (
            max_steps if max_steps is not None else self.bubble_sort_len + 2
        )
        # Proportional temperature threshold: 2 for N=4, 3 for N=5, 4 for N=6
        self.temp_threshold = max(2, array_len - 2)

        self.num_iterations = num_iterations
        self.episodes_per_iter = episodes_per_iter
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.epochs_per_iter = epochs_per_iter
        self.discount = discount
        self.device = device

        self.env = SortingProgramEnv(
            array_len=self.array_len, max_steps=self.max_steps
        )
        self.model = SortingProgramNet(
            num_test_cases=self.env.num_test_cases,
            num_actions=self.env.num_actions,
            hidden_dim=128,
            num_blocks=3,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.replay_buffer: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(
            maxlen=buffer_maxlen
        )
        self.best_program: Optional[List[Tuple[int, int]]] = None

    def _record_if_best(self, env: SortingProgramEnv) -> bool:
        """Records `env.program` if it is a solved sorting program shorter than any seen so far."""
        if not env.is_solved():
            return False
        if self.best_program is None or len(env.program) < len(self.best_program):
            self.best_program = list(env.program)
            return True
        return False

    def execute_episode(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Runs one single-player synthesis episode using MCTS.
        Stage A (in-episode): records `(obs_t, pi_t, step_reward_t)`.
        Stage B (post-episode): computes reward-to-go `z_t = G_t` backward from termination.
        """
        env = self.env.clone()
        env.reset()
        mcts = SinglePlayerMCTS(
            self.model,
            c_puct=1.5,
            discount=self.discount,
            device=self.device,
        )

        # Stage A: collect (obs_t, pi_t, step_reward_t)
        episode_history: List[Tuple[np.ndarray, np.ndarray, float]] = []
        step_idx = 0

        while not env.is_terminal():
            obs_t = env.get_obs()
            pi_t = mcts.search(
                env,
                num_simulations=self.num_simulations,
                temperature=1.0,
                add_noise=(step_idx < self.temp_threshold),
            )

            if step_idx < self.temp_threshold:
                action = int(np.random.choice(env.num_actions, p=pi_t))
            else:
                action = int(np.argmax(pi_t))

            _, reward_t, _ = env.step(action)
            episode_history.append((obs_t, pi_t, reward_t))
            step_idx += 1

        self._record_if_best(env)

        # Stage B: walk backward to compute reward-to-go z_t = G_t (Convention B state-value target)
        running_return = 0.0
        training_samples: List[Tuple[np.ndarray, np.ndarray, float]] = []

        for obs_t, pi_t, reward_t in reversed(episode_history):
            running_return = reward_t + self.discount * running_return
            training_samples.append((obs_t, pi_t, float(running_return)))

        training_samples.reverse()
        return training_samples

    def train_epoch(self) -> Tuple[float, float]:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        self.model.train()
        policy_losses, value_losses = [], []
        num_batches = max(1, len(self.replay_buffer) // self.batch_size)

        for _ in range(self.epochs_per_iter):
            for _ in range(num_batches):
                batch = random.sample(self.replay_buffer, self.batch_size)
                obs_b = torch.from_numpy(np.array([b[0] for b in batch])).to(
                    self.device
                )
                pi_b = torch.from_numpy(np.array([b[1] for b in batch])).to(
                    self.device
                )
                z_b = torch.tensor(
                    [b[2] for b in batch], dtype=torch.float32, device=self.device
                )

                logits, value = self.model(obs_b)

                log_probs = F.log_softmax(logits, dim=-1)
                policy_loss = -torch.mean(torch.sum(pi_b * log_probs, dim=-1))
                value_loss = F.mse_loss(value.squeeze(-1), z_b)

                total_loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        return float(np.mean(policy_losses)), float(np.mean(value_losses))

    def evaluate_greedy(
        self, eval_sims: Optional[int] = None
    ) -> Tuple[bool, int, int, List[Tuple[int, int]]]:
        """
        Runs a deterministic (temperature=0.0, add_noise=False) MCTS synthesis episode
        from the empty program.
        Returns `(solved, num_steps, remaining_failing_cases, program)`.
        """
        self.model.eval()
        sims = eval_sims if eval_sims is not None else self.num_simulations
        env = self.env.clone()
        env.reset()
        mcts = SinglePlayerMCTS(
            self.model, c_puct=1.5, discount=self.discount, device=self.device
        )

        while not env.is_terminal():
            pi = mcts.search(
                env, num_simulations=sims, temperature=0.0, add_noise=False
            )
            action = int(np.argmax(pi))
            env.step(action)

        self._record_if_best(env)
        return (
            env.is_solved(),
            env.steps_taken,
            env.num_failing_cases(),
            list(env.program),
        )

    def train(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path is None:
            checkpoint_path = f"sorting_n{self.array_len}_best.pt"

        print(
            f"=== Mini-AlphaDev Sorting Synthesis (N={self.array_len} indices, "
            f"{self.env.num_test_cases} binary test cases, {self.env.num_actions} swap_if_greater actions) ==="
        )
        print(
            f"Benchmarks -> Human Bubble Sort: {self.bubble_sort_len} lines | "
            f"Theoretical Optimal: {self.optimal_len} lines | "
            f"Max Steps: {self.max_steps} | Temp Threshold: {self.temp_threshold}"
        )

        best_eval_len = self.max_steps + 1

        for iteration in range(1, self.num_iterations + 1):
            for _ in tqdm(
                range(self.episodes_per_iter),
                desc=f"Iter {iteration:2d}/{self.num_iterations} [Self-Play N={self.array_len}]",
            ):
                samples = self.execute_episode()
                self.replay_buffer.extend(samples)

            p_loss, v_loss = self.train_epoch()
            solved, steps, failing, prog = self.evaluate_greedy()

            if solved and steps <= best_eval_len:
                best_eval_len = steps
                torch.save(self.model.state_dict(), checkpoint_path)
                save_str = f"★ Saved {checkpoint_path} ({steps} lines)"
            elif self.best_program is not None and best_eval_len > self.max_steps:
                torch.save(self.model.state_dict(), checkpoint_path)
                save_str = f"Saved {checkpoint_path}"
            else:
                save_str = f"Best greedy: {best_eval_len if best_eval_len <= self.max_steps else 'unsolved'}"

            status_str = (
                f"SOLVED in {steps:2d} lines"
                if solved
                else f"UNSOLVED ({failing} failing)"
            )
            best_ever_str = (
                f"{len(self.best_program)} lines"
                if self.best_program is not None
                else "none"
            )

            print(
                f"  -> Buffer: {len(self.replay_buffer):4d} | "
                f"P_Loss: {p_loss:.4f} | V_Loss: {v_loss:.4f} | "
                f"Greedy Eval: {status_str} | Best Discovered: {best_ever_str} | {save_str}"
            )

        if self.best_program is not None:
            print(
                f"\nShortest Verified Sorting Program Discovered for N={self.array_len} "
                f"({len(self.best_program)} lines vs. Bubble Sort {self.bubble_sort_len} lines):"
            )
            for idx, (i, j) in enumerate(self.best_program, 1):
                print(f"  {idx}. swap_if_greater(a, {i}, {j})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-len", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sims", type=int, default=40)
    args = parser.parse_args()

    trainer = SinglePlayerTrainer(
        array_len=args.array_len,
        num_iterations=args.iterations,
        episodes_per_iter=args.episodes,
        num_simulations=args.sims,
    )
    trainer.train()
