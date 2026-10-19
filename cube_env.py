from typing import List, Optional, Tuple
import numpy as np


# 2x2x2 Pocket Rubik's Cube Sticker Layout (24 stickers across 6 faces):
#
#           +-------+
#           | 0   1 |
#           |   U   |
#           | 2   3 |
#   +-------+-------+-------+-------+
#   | 16 17 | 8   9 | 12 13 | 20 21 |
#   |   L   |   F   |   R   |   B   |
#   | 18 19 | 10 11 | 14 15 | 22 23 |
#   +-------+-------+-------+-------+
#           | 4   5 |
#           |   D   |
#           | 6   7 |
#           +-------+
#
# By fixing the Back-Down-Left (DBL) corner cubie (stickers 6, 18, 23) in space
# as our reference frame, all 7! * 3^6 = 3,674,160 reachable states are generated
# by the 6 quarter-turn rotations of the Up (U), Front (F), and Right (R) faces:
#   0: "U"   (Up 90° clockwise)
#   1: "U'"  (Up 90° counter-clockwise)
#   2: "F"   (Front 90° clockwise)
#   3: "F'"  (Front 90° counter-clockwise)
#   4: "R"   (Right 90° clockwise)
#   5: "R'"  (Right 90° counter-clockwise)

ACTION_NAMES: Tuple[str, ...] = ("U", "U'", "F", "F'", "R", "R'")
FACE_NAMES: Tuple[str, ...] = ("U", "D", "F", "R", "L", "B")


def _build_permutation_From_cycles(cycles: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Builds a 24-element lookup table `perm` such that `new_state = state[perm]`.

    If a 4-cycle is (a, b, c, d), the sticker at `a` moves to `b`, `b` moves to `c`,
    `c` moves to `d`, and `d` moves to `a`. Therefore `new_state[b] = old_state[a]`,
    meaning `perm[b] = a`.
    """
    perm = np.arange(24, dtype=np.int64)
    for a, b, c, d in cycles:
        perm[b] = a
        perm[c] = b
        perm[d] = c
        perm[a] = d
    return perm


def _invert_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=np.int64)
    return inv


# Clockwise quarter-turn cycle definitions for U, F, R:
_PERM_U = _build_permutation_From_cycles(
    [
        (0, 1, 3, 2),      # U face clockwise
        (8, 16, 20, 12),   # F->L->B->R top-left
        (9, 17, 21, 13),   # F->L->B->R top-right
    ]
)
_PERM_U_PRIME = _invert_permutation(_PERM_U)

_PERM_F = _build_permutation_From_cycles(
    [
        (8, 9, 11, 10),    # F face clockwise
        (2, 12, 5, 19),    # U(2)->R(12)->D(5)->L(19)
        (3, 14, 4, 17),    # U(3)->R(14)->D(4)->L(17)
    ]
)
_PERM_F_PRIME = _invert_permutation(_PERM_F)

_PERM_R = _build_permutation_From_cycles(
    [
        (12, 13, 15, 14),  # R face clockwise
        (3, 20, 7, 11),    # U(3)->B(20)->D(7)->F(11)
        (1, 22, 5, 9),     # U(1)->B(22)->D(5)->F(9)
    ]
)
_PERM_R_PRIME = _invert_permutation(_PERM_R)

MOVE_PERMS: np.ndarray = np.stack(
    [_PERM_U, _PERM_U_PRIME, _PERM_F, _PERM_F_PRIME, _PERM_R, _PERM_R_PRIME],
    axis=0,
)

SOLVED_STICKERS: np.ndarray = np.repeat(np.arange(6, dtype=np.int8), 4)


class PocketCubeEnv:
    """Single-player 2x2x2 Pocket Rubik's Cube environment for AlphaZero + MCTS.

    Supports exact Reverse-Scramble Backward Curricula (`reset(scramble_depth=k)`),
    where the cube starts at the solved state and applies `k` non-canceling random
    quarter-turn moves.
    """

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self.num_actions = 6
        self.num_stickers = 24
        self.num_colors = 6
        # 24 * 6 = 144 one-hot sticker features + 1 normalized step counter = 145
        self.obs_dim = self.num_stickers * self.num_colors + 1

        self.stickers: np.ndarray = SOLVED_STICKERS.copy()
        self.steps_taken: int = 0
        self.last_action: Optional[int] = None
        self.move_history: List[int] = []
        self.scramble_moves: List[int] = []

    def reset(
        self,
        scramble_depth: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Resets the cube to the solved state and applies `scramble_depth` random moves."""
        self.stickers = SOLVED_STICKERS.copy()
        self.steps_taken = 0
        self.last_action = None
        self.move_history = []
        self.scramble_moves = []

        if scramble_depth > 0:
            self.scramble(scramble_depth, rng=rng)
        return self.get_observation()

    def scramble(
        self,
        scramble_depth: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        """Applies `scramble_depth` random moves from the solved state without
        immediate inverse cancellations (e.g., U then U') or 3 identical turns in a row
        (e.g., U, U, U which reduces to U').
        """
        if rng is None:
            rng = np.random.default_rng()

        self.stickers = SOLVED_STICKERS.copy()
        self.steps_taken = 0
        self.move_history = []
        self.scramble_moves = []

        prev_action: Optional[int] = None
        prev_prev_action: Optional[int] = None

        for _ in range(scramble_depth):
            candidates = []
            for a in range(self.num_actions):
                # Never immediately undo the previous scramble move
                if prev_action is not None and a == (prev_action ^ 1):
                    continue
                # Never turn the same face 3 times in the same direction (or 3 times on same face)
                if (
                    prev_action is not None
                    and prev_prev_action is not None
                    and (a // 2) == (prev_action // 2) == (prev_prev_action // 2)
                ):
                    continue
                candidates.append(a)

            chosen = int(rng.choice(candidates))
            self.stickers = self.stickers[MOVE_PERMS[chosen]]
            self.scramble_moves.append(chosen)
            prev_prev_action = prev_action
            prev_action = chosen

        # If a short random walk happened to loop back to solved, nudge by 1 move
        if self.is_solved() and scramble_depth > 0:
            chosen = 0 if prev_action != 1 else 2
            self.stickers = self.stickers[MOVE_PERMS[chosen]]
            self.scramble_moves.append(chosen)

        self.last_action = None
        return self.scramble_moves

    def clone(self) -> "PocketCubeEnv":
        """Fast shallow copy of the environment for MCTS simulation."""
        copy_env = PocketCubeEnv.__new__(PocketCubeEnv)
        copy_env.max_steps = self.max_steps
        copy_env.num_actions = self.num_actions
        copy_env.num_stickers = self.num_stickers
        copy_env.num_colors = self.num_colors
        copy_env.obs_dim = self.obs_dim
        copy_env.stickers = self.stickers.copy()
        copy_env.steps_taken = self.steps_taken
        copy_env.last_action = self.last_action
        copy_env.move_history = self.move_history.copy()
        copy_env.scramble_moves = self.scramble_moves
        return copy_env

    def is_solved(self) -> bool:
        """Returns True iff every face is monochromatic."""
        # Because the DBL corner (stickers 6, 18, 23) is fixed in space,
        # the only monochromatic configuration is the exact SOLVED_STICKERS array.
        return bool(np.array_equal(self.stickers, SOLVED_STICKERS))

    def is_terminal(self) -> bool:
        return self.is_solved() or (self.steps_taken >= self.max_steps)

    def legal_actions(self) -> List[int]:
        """Returns legal quarter-turn actions, pruning the immediate inverse `last_action ^ 1`."""
        if self.is_terminal():
            return []
        if self.last_action is None:
            return [0, 1, 2, 3, 4, 5]
        forbidden = self.last_action ^ 1
        return [a for a in range(self.num_actions) if a != forbidden]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Applies quarter-turn `action` in {0..5}.

        Reward structure (sparse combinatorial discovery):
        - Every move costs -1.0 step penalty.
        - If the episode hits `max_steps` without solving the cube, an additional
          timeout penalty of `-self.max_steps` is applied so the agent strictly prefers
          solving the cube over timing out.
        """
        if self.steps_taken >= self.max_steps:
            raise RuntimeError("Cannot call step() past max_steps on PocketCubeEnv.")

        self.stickers = self.stickers[MOVE_PERMS[action]]
        self.steps_taken += 1
        self.last_action = action
        self.move_history.append(action)

        solved = self.is_solved()
        done = solved or (self.steps_taken >= self.max_steps)

        reward = -1.0
        if done and not solved:
            reward -= float(self.max_steps)

        return self.get_observation(), reward, done

    def get_observation(self) -> np.ndarray:
        """Returns a (145,) float32 vector:
        - `obs[:144]`: 24 x 6 one-hot encoding of sticker colors
        - `obs[144]`: normalized `steps_taken / max_steps`
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        flat_indices = np.arange(24) * 6 + self.stickers
        obs[flat_indices] = 1.0
        obs[-1] = float(self.steps_taken) / float(self.max_steps)
        return obs

    get_obs = get_observation
