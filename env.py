from typing import List, Tuple
import numpy as np


class SortingProgramEnv:
    """
    Single-player AlphaZero environment (Mini-AlphaDev) for synthesizing
    fixed-length sorting programs out of `swap_if_greater(a, i, j)` statements.

    Terminology:
    - `array_len` (N): Number of elements/indices in the array `a[0..N-1]`.
    - Action `a`: Index into `self.swap_pairs`, representing one statement:
          `swap_if_greater(a, i, j)`  (with 0 <= i < j < N)
    - State (`obs`): Binary float32 vector of shape `(2^N,)` where `obs[c] = 1.0`
      iff binary test case `c` in `{0, 1}^N` is still reachable and NOT yet sorted.
    - Solved: When `0` failing binary test cases remain (`np.sum(obs) == 0`).
      By Knuth's Zero-One Principle, a program of `swap_if_greater` statements that
      sorts all `2^N` binary arrays is guaranteed to sort any array of N numbers.
    """

    def __init__(self, array_len: int = 4, max_steps: int = 10):
        self.array_len = array_len
        self.num_test_cases = 1 << array_len  # 2^N binary test cases
        self.obs_dim = self.num_test_cases + 1  # (2^N failing test cases) + (steps_taken / max_steps)
        self.max_steps = max_steps

        # All N*(N-1)//2 index pairs (i, j) with i < j
        self.swap_pairs: List[Tuple[int, int]] = [
            (i, j)
            for i in range(array_len)
            for j in range(i + 1, array_len)
        ]
        self.num_actions = len(self.swap_pairs)

        # Precompute lookup tables once so step() and legal_actions() are O(1) vectorized ops
        self._build_lookup_tables()

        # Mutable episode state
        self.failing_mask: np.ndarray = np.zeros(self.num_test_cases, dtype=bool)
        self.program: List[Tuple[int, int]] = []
        self.steps_taken: int = 0

        self.reset()

    def _build_lookup_tables(self) -> None:
        """
        Precomputes:
        - `self.initial_failing_mask`: shape (2^N,), True for all unsorted binary arrays.
        - `self.transition_table`: shape (num_actions, 2^N), maps (action, case_idx) -> new_case_idx
          after executing `swap_if_greater(a, i, j)`.
        - `self.causes_swap_mask`: shape (num_actions, 2^N), True iff `swap_if_greater(a, i, j)`
          actually swaps `a[i]=1` and `a[j]=0` on binary test case `c`.
        """
        N = self.array_len
        num_cases = self.num_test_cases

        # Decode all 2^N integers into binary arrays a[0..N-1] of shape (2^N, N)
        case_ids = np.arange(num_cases, dtype=np.int32)
        bits = ((case_ids[:, None] >> np.arange(N)) & 1).astype(np.int32)

        # A binary array is sorted iff a[0] <= a[1] <= ... <= a[N-1]
        is_sorted = np.all(bits[:, :-1] <= bits[:, 1:], axis=1)
        self.is_sorted_mask = is_sorted
        self.initial_failing_mask = ~is_sorted

        self.transition_table = np.zeros((self.num_actions, num_cases), dtype=np.int32)
        self.causes_swap_mask = np.zeros((self.num_actions, num_cases), dtype=bool)

        for act_idx, (i, j) in enumerate(self.swap_pairs):
            # swap_if_greater(a, i, j) swaps iff a[i] == 1 and a[j] == 0
            needs_swap = (bits[:, i] == 1) & (bits[:, j] == 0)
            self.causes_swap_mask[act_idx] = needs_swap

            # When a[i]=1 and a[j]=0 swap, bit i clears and bit j sets
            swapped_case_ids = case_ids ^ ((1 << i) | (1 << j))
            self.transition_table[act_idx] = np.where(
                needs_swap, swapped_case_ids, case_ids
            )

    def reset(self) -> np.ndarray:
        """Resets to an empty program with all unsorted binary test cases active."""
        self.failing_mask = self.initial_failing_mask.copy()
        self.program = []
        self.steps_taken = 0
        return self.get_obs()

    def clone(self) -> "SortingProgramEnv":
        """Fast shallow copy of episode state sharing static lookup tables."""
        cloned = SortingProgramEnv.__new__(SortingProgramEnv)
        cloned.array_len = self.array_len
        cloned.num_test_cases = self.num_test_cases
        cloned.obs_dim = self.obs_dim
        cloned.max_steps = self.max_steps
        cloned.swap_pairs = self.swap_pairs
        cloned.num_actions = self.num_actions
        cloned.is_sorted_mask = self.is_sorted_mask
        cloned.initial_failing_mask = self.initial_failing_mask
        cloned.transition_table = self.transition_table
        cloned.causes_swap_mask = self.causes_swap_mask

        cloned.failing_mask = self.failing_mask.copy()
        cloned.program = list(self.program)
        cloned.steps_taken = self.steps_taken
        return cloned

    def get_obs(self) -> np.ndarray:
        """
        Returns float32 observation vector of shape (2^N + 1,):
        - `obs[:-1]`: binary vector of shape (2^N,) indicating active failing test cases.
        - `obs[-1]`: normalized step progress `steps_taken / max_steps` in [0.0, 1.0].
        """
        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[:-1] = self.failing_mask.astype(np.float32)
        obs[-1] = float(self.steps_taken) / float(self.max_steps)
        return obs

    def num_failing_cases(self) -> int:
        """Returns count of binary test cases that are still unsorted."""
        return int(np.count_nonzero(self.failing_mask))

    def is_solved(self) -> bool:
        """True iff 0 failing binary test cases remain (program sorts all inputs)."""
        return not np.any(self.failing_mask)

    def is_terminal(self) -> bool:
        """True if all test cases are sorted OR step limit is reached."""
        return self.is_solved() or (self.steps_taken >= self.max_steps)

    def legal_actions(self) -> List[int]:
        """
        Returns action indices for `swap_if_greater(a, i, j)` statements that change
        at least one currently failing test case (pruning pure no-ops).
        """
        if self.is_terminal():
            return []
        # An action is useful iff it swaps a[i]>a[j] on at least one active failing test case
        useful_mask = np.any(self.causes_swap_mask[:, self.failing_mask], axis=1)
        return np.flatnonzero(useful_mask).tolist()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Appends `swap_if_greater(a, i, j)` to the program, updates the failing test cases,
        and returns `(obs, reward, done)`.

        Reward structure:
        - Each statement costs `-1.0` (encouraging the shortest valid program).
        - If `steps_taken == max_steps` and the program is STILL not solved, an additional
          graded penalty `-num_failing_cases()` is applied so programs that solve more test
          cases receive a warmer value signal than programs that solve fewer.
        """
        if self.is_terminal():
            raise RuntimeError("Cannot call step() on a terminal SortingProgramEnv.")

        self.program.append(self.swap_pairs[action])
        self.steps_taken += 1

        # Map all currently failing test cases through transition_table[action]
        active_cases = np.flatnonzero(self.failing_mask)
        next_cases = self.transition_table[action, active_cases]

        new_failing = np.zeros(self.num_test_cases, dtype=bool)
        new_failing[next_cases] = True
        # Clear any test cases that have now become sorted!
        new_failing[self.is_sorted_mask] = False
        self.failing_mask = new_failing

        reward = -1.0
        done = self.is_terminal()

        if done and not self.is_solved():
            # Graded terminal penalty proportional to remaining failing test cases
            reward -= float(self.num_failing_cases())

        return self.get_obs(), reward, done

    def format_program(self) -> str:
        """Returns human-readable code listing of the current program."""
        if not self.program:
            return "# <empty program>"
        lines = [
            f"  {step + 1}. swap_if_greater(a, {i}, {j})"
            for step, (i, j) in enumerate(self.program)
        ]
        return "\n".join(lines)
