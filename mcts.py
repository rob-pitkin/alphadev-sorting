import math
from typing import Dict, Iterator, Optional, Tuple
import numpy as np
import torch

from env import SortingProgramEnv


class MinMaxStats:
    """
    Tracks the running [minimum, maximum] of action-values Q(s, a) across all
    visited action edges in the current MCTS search tree, mapping Q into [0, 1].
    """

    def __init__(self) -> None:
        self.maximum: float = -float("inf")
        self.minimum: float = float("inf")

    def update(self, value: float) -> None:
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        """
        Normalizes `value` into [0, 1] when `maximum > minimum`.
        In the degenerate case (`maximum <= minimum`, e.g., only 1 child visited
        or all visited children tied), returns `0.0` (matching unvisited siblings)
        so MCTS naturally explores additional branches to establish a spread.
        """
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return 0.0


class Node:
    """
    A single state node in the single-player MCTS tree (DeepMind Convention B).
    - `node.value` stores the state-value V(s_node): expected return from `node` onward.
    - The action-value Q(s, a) of the edge into `child` is:
          Q(s, a) = child.reward + discount * child.value
    """

    def __init__(
        self,
        prior: float,
        parent: Optional["Node"] = None,
        action: Optional[int] = None,
    ):
        self.prior: float = prior
        self.parent: Optional["Node"] = parent
        self.action: Optional[int] = action

        self.children: Dict[int, "Node"] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.reward: float = 0.0  # Immediate step reward r(s_parent, action)

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        """Expected future return V(s) from this state onward."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def child_q_value(self, child: "Node", discount: float = 1.0) -> float:
        """Computes action-value Q(s, a) = r(s, a) + discount * V(s') for an edge."""
        if child.visit_count == 0:
            return 0.0
        return child.reward + discount * child.value

    def puct_score(
        self,
        child: "Node",
        c_puct: float,
        discount: float,
        min_max_stats: MinMaxStats,
    ) -> float:
        """
        Single-player PUCT with Min-Max Q-Normalization into [0, 1] (Convention B):
        - Visited child (N > 0):
              q = child.reward + discount * child.value
              q_norm = min_max_stats.normalize(q)
        - Unvisited child (N == 0):
              q_norm = 0.0
        """
        if child.visit_count > 0:
            q = self.child_q_value(child, discount)
            q_norm = min_max_stats.normalize(q)
        else:
            q_norm = 0.0

        u = (
            c_puct
            * child.prior
            * (math.sqrt(self.visit_count) / (1 + child.visit_count))
        )
        return q_norm + u


class SinglePlayerMCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        c_puct: float = 1.5,
        discount: float = 1.0,
        dirichlet_alpha: float = 0.5,
        dirichlet_eps: float = 0.25,
        device: str = "cpu",
    ):
        self.model = model
        self.c_puct = c_puct
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device

    def select(
        self, root: Node, scratch_env: SortingProgramEnv, min_max_stats: MinMaxStats
    ) -> Node:
        """
        Traverses from `root` down the tree using normalized PUCT until reaching
        an unexpanded leaf node or a terminal environment state.
        Records the immediate step reward on each child node along the way.
        """
        node = root
        while node.is_expanded and not scratch_env.is_terminal():
            best_action = -1
            best_score = -float("inf")
            best_child = None

            for action, child in node.children.items():
                score = node.puct_score(
                    child, self.c_puct, self.discount, min_max_stats
                )
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child = child

            _, step_reward, _ = scratch_env.step(best_action)
            best_child.reward = step_reward
            node = best_child

        return node

    @torch.no_grad()
    def expand(self, leaf_node: Node, scratch_env: SortingProgramEnv) -> float:
        """
        Evaluates `leaf_node` and populates `leaf_node.children` for legal actions.
        Returns the estimated remaining value V(s_leaf) AFTER `leaf_node`:
        - If `scratch_env.is_terminal()`, 0.0 remaining steps/rewards occur after `leaf_node`
          (the terminal step reward is already stored in `leaf_node.reward`!).
        - Otherwise, evaluates `self.model(obs)` to get `(logits, v_leaf)`.
        """
        if scratch_env.is_terminal():
            return 0.0

        obs = scratch_env.get_obs()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        logits, value = self.model(obs_tensor)
        logits_1d = logits.squeeze(0)  # (num_actions,)
        v_leaf = float(value.item())

        legal_actions = scratch_env.legal_actions()
        masked_logits = torch.full_like(logits_1d, -float("inf"))
        masked_logits[legal_actions] = logits_1d[legal_actions]
        priors = torch.softmax(masked_logits, dim=-1).cpu().numpy()

        for action in legal_actions:
            leaf_node.children[action] = Node(
                prior=float(priors[action]), parent=leaf_node, action=action
            )

        return v_leaf

    def backpropagate(
        self, leaf_node: Node, leaf_value: float, min_max_stats: MinMaxStats
    ) -> None:
        """
        Walks from `leaf_node` up to the root accumulating state values V(s_node) (Convention B):
        1. Adds `cumulative_return` (return from `node` onward) to `node.value_sum`.
        2. Computes the incoming edge action-value `q = node.reward + discount * node.value`
           and updates `min_max_stats`.
        3. Steps `cumulative_return = node.reward + discount * cumulative_return` for `node.parent`.
        """
        cumulative_return = leaf_value
        node: Optional[Node] = leaf_node

        while node is not None:
            node.visit_count += 1
            node.value_sum += cumulative_return

            if node.parent is not None:
                edge_q = node.reward + self.discount * node.value
                min_max_stats.update(edge_q)

            cumulative_return = node.reward + self.discount * cumulative_return
            node = node.parent

    def apply_dirichlet_noise(self, root: Node) -> None:
        if not root.children:
            return
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, n_val in zip(actions, noise):
            child = root.children[action]
            child.prior = (
                1.0 - self.dirichlet_eps
            ) * child.prior + self.dirichlet_eps * float(n_val)

    def get_action_probs(
        self, root: Node, num_actions: int, temperature: float = 1.0
    ) -> np.ndarray:
        visits = np.zeros(num_actions, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = float(child.visit_count)

        if temperature == 0.0:
            best_action = int(np.argmax(visits))
            probs = np.zeros(num_actions, dtype=np.float32)
            probs[best_action] = 1.0
            return probs

        scaled = np.power(visits, 1.0 / temperature)
        total = float(np.sum(scaled))
        if total == 0.0:
            return visits
        return scaled / total

    def search_step_by_step(
        self,
        root_env: SortingProgramEnv,
        num_simulations: int,
        add_noise: bool = False,
    ) -> Iterator[Tuple[int, Node, float, MinMaxStats]]:
        """
        Generator yielding `(sim_idx, root, root_net_value, min_max_stats)` after each simulation
        so interactive visualizers can animate live MCTS search dynamics.
        """
        self.model.eval()
        min_max_stats = MinMaxStats()
        root = Node(prior=1.0, parent=None, action=None)

        root_net_value = self.expand(root, root_env)
        root.visit_count = 1
        root.value_sum = root_net_value

        if add_noise:
            self.apply_dirichlet_noise(root)

        for sim_idx in range(1, num_simulations + 1):
            scratch_env = root_env.clone()
            leaf = self.select(root, scratch_env, min_max_stats)
            leaf_value = self.expand(leaf, scratch_env)
            self.backpropagate(leaf, leaf_value, min_max_stats)
            yield sim_idx, root, root_net_value, min_max_stats

    def search(
        self,
        root_env: SortingProgramEnv,
        num_simulations: int,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        root = None
        for _, root, _, _ in self.search_step_by_step(
            root_env, num_simulations, add_noise=add_noise
        ):
            pass
        assert root is not None
        return self.get_action_probs(root, root_env.num_actions, temperature)
