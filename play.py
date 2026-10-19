import os
from typing import List, Optional, Tuple
import numpy as np
import pygame
import torch

from env import SortingProgramEnv
from mcts import MinMaxStats, Node, SinglePlayerMCTS
from model import SortingProgramNet
from train import OPTIMAL_LENGTHS


def get_bubble_sort_schedule(array_len: int) -> List[Tuple[int, int]]:
    """Returns the standard human Bubble Sort sequence of swap_if_greater(i, j) pairs."""
    seq = []
    for end in range(array_len - 1, 0, -1):
        for i in range(end):
            seq.append((i, i + 1))
    return seq


class SortingProgramVisualizer:
    def __init__(self, array_len: int = 5, num_simulations: int = 120):
        pygame.init()
        self.width = 1180
        self.height = 820
        self.screen = pygame.display.set_caption(
            "Mini-AlphaDev — Single-Player AlphaZero Sorting Program Synthesizer"
        )
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("menlo", 17, bold=True)
        self.font_md = pygame.font.SysFont("menlo", 14, bold=True)
        self.font_sm = pygame.font.SysFont("menlo", 12)
        self.font_xs = pygame.font.SysFont("menlo", 11)

        self.num_simulations = num_simulations
        self.auto_play = False
        self.action_rects: List[pygame.Rect] = []

        self.load_problem_size(array_len)

    def load_problem_size(self, array_len: int) -> None:
        self.array_len = array_len
        self.bubble_schedule = get_bubble_sort_schedule(array_len)
        self.bubble_len = len(self.bubble_schedule)
        self.optimal_len = OPTIMAL_LENGTHS.get(array_len, self.bubble_len)
        self.max_steps = self.bubble_len + 2

        self.env = SortingProgramEnv(
            array_len=self.array_len, max_steps=self.max_steps
        )
        self.model = SortingProgramNet(
            num_test_cases=self.env.num_test_cases,
            num_actions=self.env.num_actions,
            hidden_dim=128,
            num_blocks=3,
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt = os.path.join(base_dir, f"sorting_n{array_len}_best.pt")
        self.ckpt_loaded = False
        if os.path.exists(ckpt):
            state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            self.ckpt_loaded = True
        self.model.eval()

        self.mcts = SinglePlayerMCTS(self.model, c_puct=1.5, discount=1.0)
        self.history_stack: List[SortingProgramEnv] = []
        self.reset_episode()

    def reset_episode(self) -> None:
        self.env.reset()
        self.history_stack = []
        self.auto_play = False
        self.search_gen = None
        self.sim_progress = 0
        self._refresh_static_evaluation()

    def undo_step(self) -> None:
        if self.history_stack:
            self.env = self.history_stack.pop()
            self.auto_play = False
            self.search_gen = None
            self.sim_progress = 0
            self._refresh_static_evaluation()

    @torch.no_grad()
    def _refresh_static_evaluation(self) -> None:
        """Computes raw network prior P(s, a) and V(s) on the current state before MCTS runs."""
        num_a = self.env.num_actions
        self.priors = np.zeros(num_a, dtype=np.float32)
        self.visit_probs = np.zeros(num_a, dtype=np.float32)
        self.raw_q = np.full(num_a, np.nan, dtype=np.float32)
        self.norm_q = np.zeros(num_a, dtype=np.float32)
        self.q_min = float("inf")
        self.q_max = -float("inf")

        if self.env.is_terminal():
            self.net_v = 0.0
            self.mcts_v = 0.0
            return

        obs_t = torch.from_numpy(self.env.get_obs()).unsqueeze(0)
        logits, val = self.model(obs_t)
        logits_1d = logits.squeeze(0)
        self.net_v = float(val.item())
        self.mcts_v = self.net_v

        legal = self.env.legal_actions()
        masked = torch.full_like(logits_1d, -float("inf"))
        masked[legal] = logits_1d[legal]
        self.priors = torch.softmax(masked, dim=-1).cpu().numpy()

    def start_mcts_step(self) -> None:
        if self.env.is_terminal() or self.search_gen is not None:
            return
        self.search_gen = self.mcts.search_step_by_step(
            self.env, self.num_simulations, add_noise=False
        )
        self.sim_progress = 0

    def update_mcts_animation(self) -> None:
        if self.search_gen is None:
            if self.auto_play and not self.env.is_terminal():
                self.start_mcts_step()
            return

        # Advance a few simulations per frame for smooth 60 FPS animation
        sims_per_frame = max(1, self.num_simulations // 22)
        root: Optional[Node] = None
        min_max_stats: Optional[MinMaxStats] = None
        finished = False

        for _ in range(sims_per_frame):
            try:
                sim_idx, root, net_v, min_max_stats = next(self.search_gen)
                self.sim_progress = sim_idx
                self.net_v = net_v
            except StopIteration:
                finished = True
                break

        if root is not None and min_max_stats is not None:
            self._extract_root_telemetry(root, min_max_stats)

        if finished:
            self.search_gen = None
            best_action = int(np.argmax(self.visit_probs))
            self.apply_user_or_ai_action(best_action)

    def _extract_root_telemetry(
        self, root: Node, min_max_stats: MinMaxStats
    ) -> None:
        num_a = self.env.num_actions
        visits = np.zeros(num_a, dtype=np.float32)
        self.raw_q = np.full(num_a, np.nan, dtype=np.float32)
        self.norm_q = np.zeros(num_a, dtype=np.float32)

        for action, child in root.children.items():
            self.priors[action] = child.prior
            visits[action] = float(child.visit_count)
            if child.visit_count > 0:
                q = root.child_q_value(child, discount=1.0)
                self.raw_q[action] = q
                self.norm_q[action] = min_max_stats.normalize(q)

        total_v = float(np.sum(visits))
        self.visit_probs = visits / total_v if total_v > 0 else visits
        self.mcts_v = root.value
        self.q_min = min_max_stats.minimum
        self.q_max = min_max_stats.maximum

    def apply_user_or_ai_action(self, action: int) -> None:
        if self.env.is_terminal():
            return
        self.history_stack.append(self.env.clone())
        self.env.step(action)
        self.search_gen = None
        self.sim_progress = 0
        self._refresh_static_evaluation()

    def step_bubble_sort(self) -> None:
        """Applies the next statement in Human Bubble Sort for side-by-side comparison."""
        if self.env.is_terminal():
            return
        step_idx = len(self.env.program)
        if step_idx < len(self.bubble_schedule):
            pair = self.bubble_schedule[step_idx]
            action = self.env.swap_pairs.index(pair)
            self.apply_user_or_ai_action(action)

    def draw(self, hovered_action: Optional[int]) -> None:
        self.screen.fill((16, 20, 28))

        # 1. Top Header Bar
        self._draw_header()

        # 2. Top-Left Panel: MCTS Action Telemetry Bars
        self._draw_telemetry_panel(hovered_action)

        # 3. Top-Right Panel: Value Head & MinMaxStats Gauge
        self._draw_value_panel()

        # 4. Middle Panel: Array Index Timeline & swap_if_greater(a, i, j) Connectors
        self._draw_program_timeline(hovered_action)

        # 5. Bottom Panel: Live Remaining Failing Binary Test Cases
        self._draw_failing_cases_panel()

        pygame.display.flip()

    def _draw_header(self) -> None:
        header_rect = pygame.Rect(16, 12, self.width - 32, 64)
        pygame.draw.rect(self.screen, (26, 32, 44), header_rect, border_radius=8)
        pygame.draw.rect(
            self.screen, (52, 65, 88), header_rect, width=1, border_radius=8
        )

        ckpt_tag = (
            f"sorting_n{self.array_len}_best.pt"
            if self.ckpt_loaded
            else "Untrained"
        )
        title_str = (
            f"Mini-AlphaDev: N={self.array_len} Array Indices  |  "
            f"Lines: {self.env.steps_taken}/{self.max_steps}  |  "
            f"Bubble Sort: {self.bubble_len} lines  |  Optimal: {self.optimal_len} lines"
        )
        title_surf = self.font_title.render(title_str, True, (235, 242, 255))
        self.screen.blit(title_surf, (28, 19))

        ctrl_str = (
            f"[{ckpt_tag}]  [Space] AI Step  [A] Auto ({'ON' if self.auto_play else 'OFF'})  "
            f"[Click] Manual Step  [B] Bubble Step  [U] Undo  [R] Reset  [4/5/6] Change N  "
            f"[Up/Down] Sims: {self.num_simulations}"
        )
        ctrl_surf = self.font_xs.render(ctrl_str, True, (158, 175, 205))
        self.screen.blit(ctrl_surf, (28, 46))

    def _draw_telemetry_panel(self, hovered_action: Optional[int]) -> None:
        panel = pygame.Rect(16, 86, 864, 250)
        pygame.draw.rect(self.screen, (22, 28, 38), panel, border_radius=8)
        pygame.draw.rect(self.screen, (45, 58, 78), panel, width=1, border_radius=8)
        self.screen.set_clip(panel)

        hdr = self.font_md.render(
            "Action Telemetry — Blue: Prior P(s,a)  |  Green: Visits π(a|s)",
            True,
            (215, 228, 248),
        )
        self.screen.blit(hdr, (28, 96))

        status = (
            f"Searching: {self.sim_progress}/{self.num_simulations} sims"
            if self.search_gen is not None
            else "Ready ([Space] MCTS / Click column)"
        )
        status_col = (
            (110, 230, 165) if self.search_gen is not None else (150, 170, 200)
        )
        status_surf = self.font_sm.render(status, True, status_col)
        self.screen.blit(
            status_surf, (panel.right - status_surf.get_width() - 16, 98)
        )

        num_a = self.env.num_actions
        legal_set = set(self.env.legal_actions())
        self.action_rects = []

        margin_x = 20
        usable_w = panel.width - 2 * margin_x
        slot_w = usable_w / num_a
        bar_max_h = 130
        base_y = panel.y + 185

        scale_denom = max(
            0.35, float(np.max(self.priors)), float(np.max(self.visit_probs))
        )

        for a_idx, (i, j) in enumerate(self.env.swap_pairs):
            slot_x = panel.x + margin_x + int(a_idx * slot_w)
            col_rect = pygame.Rect(slot_x + 2, panel.y + 34, int(slot_w) - 4, 208)
            self.action_rects.append(col_rect)

            is_legal = a_idx in legal_set
            is_hovered = hovered_action == a_idx and is_legal

            bg_col = (
                (38, 52, 74)
                if is_hovered
                else ((28, 35, 48) if is_legal else (20, 23, 30))
            )
            pygame.draw.rect(self.screen, bg_col, col_rect, border_radius=5)

            # Prior bar (Blue)
            p_val = float(self.priors[a_idx])
            p_h = int((p_val / scale_denom) * bar_max_h)
            bar_w = max(8, int(slot_w * 0.34))
            p_rect = pygame.Rect(
                col_rect.centerx - bar_w - 2, base_y - p_h, bar_w, p_h
            )
            pygame.draw.rect(self.screen, (66, 145, 245), p_rect, border_radius=3)

            # Visits bar (Green)
            v_val = float(self.visit_probs[a_idx])
            v_h = int((v_val / scale_denom) * bar_max_h)
            v_rect = pygame.Rect(col_rect.centerx + 2, base_y - v_h, bar_w, v_h)
            pygame.draw.rect(self.screen, (56, 215, 135), v_rect, border_radius=3)

            # Action label swap(i, j)
            lbl_col = (240, 245, 255) if is_legal else (95, 105, 120)
            lbl = self.font_sm.render(f"({i},{j})", True, lbl_col)
            self.screen.blit(lbl, (col_rect.centerx - lbl.get_width() // 2, base_y + 5))

            if not is_legal and not self.env.is_terminal():
                noop_s = self.font_xs.render("no-op", True, (140, 90, 90))
                self.screen.blit(
                    noop_s, (col_rect.centerx - noop_s.get_width() // 2, base_y + 23)
                )
            elif not np.isnan(self.raw_q[a_idx]):
                q_s = self.font_xs.render(
                    f"Q:{self.raw_q[a_idx]:.1f}", True, (185, 200, 225)
                )
                qn_s = self.font_xs.render(
                    f"Q̄:{self.norm_q[a_idx]:.2f}", True, (120, 225, 170)
                )
                self.screen.blit(
                    q_s, (col_rect.centerx - q_s.get_width() // 2, base_y + 21)
                )
                self.screen.blit(
                    qn_s, (col_rect.centerx - qn_s.get_width() // 2, base_y + 35)
                )

        self.screen.set_clip(None)

    def _draw_value_panel(self) -> None:
        panel = pygame.Rect(892, 86, self.width - 908, 250)
        pygame.draw.rect(self.screen, (22, 28, 38), panel, border_radius=8)
        pygame.draw.rect(self.screen, (45, 58, 78), panel, width=1, border_radius=8)
        self.screen.set_clip(panel)

        hdr = self.font_md.render("Value & MinMaxStats", True, (215, 228, 248))
        self.screen.blit(hdr, (panel.x + 14, panel.y + 10))

        lines = [
            (
                f"Net V(s):  {self.net_v:6.2f} lines",
                (90, 170, 255),
                "System 1 (-softplus)",
            ),
            (
                f"MCTS V(s): {self.mcts_v:6.2f} lines",
                (70, 225, 145),
                "System 2 lookahead",
            ),
        ]
        if self.q_max >= self.q_min:
            lines.append(
                (
                    f"Tree Q_min: {self.q_min:6.2f}",
                    (230, 140, 110),
                    "Maps to Q̄ = 0.00",
                )
            )
            lines.append(
                (
                    f"Tree Q_max: {self.q_max:6.2f}",
                    (120, 235, 165),
                    "Maps to Q̄ = 1.00",
                )
            )
        else:
            lines.append(
                ("Tree [Qmin, Qmax]:", (150, 165, 190), "Press [Space] to run MCTS")
            )

        y = panel.y + 42
        for main_txt, color, sub_txt in lines:
            s1 = self.font_md.render(main_txt, True, color)
            s2 = self.font_xs.render(sub_txt, True, (135, 150, 175))
            self.screen.blit(s1, (panel.x + 14, y))
            self.screen.blit(s2, (panel.x + 14, y + 19))
            y += 48

        self.screen.set_clip(None)

    def _draw_program_timeline(self, hovered_action: Optional[int]) -> None:
        panel = pygame.Rect(16, 348, self.width - 32, 245)
        pygame.draw.rect(self.screen, (22, 28, 38), panel, border_radius=8)
        pygame.draw.rect(self.screen, (45, 58, 78), panel, width=1, border_radius=8)
        self.screen.set_clip(panel)

        if self.env.is_solved():
            diff = self.bubble_len - self.env.steps_taken
            badge = (
                f"SOLVED in {self.env.steps_taken} lines! "
                f"({diff:+d} lines shorter than Bubble Sort!)"
                if diff > 0
                else f"SOLVED in {self.env.steps_taken} lines!"
            )
            hdr_col = (75, 235, 145)
        elif self.env.is_terminal():
            badge = f"TIMED OUT at {self.env.steps_taken} lines ({self.env.num_failing_cases()} failing cases left)"
            hdr_col = (245, 105, 105)
        else:
            badge = "Program Execution Timeline — Vertical connectors show swap_if_greater(a, i, j) lines"
            hdr_col = (215, 228, 248)

        hdr = self.font_md.render(badge, True, hdr_col)
        self.screen.blit(hdr, (28, 358))

        N = self.array_len
        top_y = panel.y + 52
        bot_y = panel.y + panel.height - 28
        wire_ys = [
            int(top_y + idx * (bot_y - top_y) / max(1, N - 1)) for idx in range(N)
        ]

        start_x = panel.x + 95
        end_x = panel.x + panel.width - 40
        step_slots = max(self.max_steps, len(self.env.program) + 1)
        slot_dx = (end_x - start_x) / max(1, step_slots)

        # Draw horizontal array index lines a[0] .. a[N-1]
        for idx, wy in enumerate(wire_ys):
            lbl = self.font_md.render(f"a[{idx}]", True, (195, 210, 235))
            self.screen.blit(lbl, (panel.x + 28, wy - 9))
            pygame.draw.line(
                self.screen, (75, 92, 120), (start_x - 15, wy), (end_x, wy), width=2
            )

        # Draw placed swap_if_greater(a, i, j) statements
        for s_idx, (i, j) in enumerate(self.env.program):
            cx = int(start_x + (s_idx + 0.5) * slot_dx)
            y1, y2 = wire_ys[i], wire_ys[j]
            pygame.draw.line(
                self.screen, (80, 220, 145), (cx, y1), (cx, y2), width=4
            )
            pygame.draw.circle(self.screen, (110, 250, 170), (cx, y1), 7)
            pygame.draw.circle(self.screen, (110, 250, 170), (cx, y2), 7)

            step_lbl = self.font_xs.render(
                f"#{s_idx+1}:({i},{j})", True, (175, 235, 200)
            )
            self.screen.blit(
                step_lbl, (cx - step_lbl.get_width() // 2, panel.y + panel.height - 20)
            )

        # Draw ghost preview of hovered action
        if hovered_action is not None and not self.env.is_terminal():
            i, j = self.env.swap_pairs[hovered_action]
            cx = int(start_x + (len(self.env.program) + 0.5) * slot_dx)
            y1, y2 = wire_ys[i], wire_ys[j]
            pygame.draw.line(
                self.screen, (95, 165, 255), (cx, y1), (cx, y2), width=2
            )
            pygame.draw.circle(self.screen, (95, 165, 255), (cx, y1), 6, width=2)
            pygame.draw.circle(self.screen, (95, 165, 255), (cx, y2), 6, width=2)

        self.screen.set_clip(None)

    def _draw_failing_cases_panel(self) -> None:
        panel = pygame.Rect(16, 604, self.width - 32, 202)
        pygame.draw.rect(self.screen, (22, 28, 38), panel, border_radius=8)
        pygame.draw.rect(self.screen, (45, 58, 78), panel, width=1, border_radius=8)
        self.screen.set_clip(panel)

        failing_indices = np.flatnonzero(self.env.failing_mask)
        count_f = len(failing_indices)
        hdr_txt = f"Active Failing Binary Test Cases: {count_f} / {self.env.num_test_cases} remaining"
        hdr = self.font_md.render(hdr_txt, True, (215, 228, 248))
        self.screen.blit(hdr, (28, 614))

        sub_txt = "Knuth's 0-1 Principle: 0 remaining => Sorts ALL real arrays!"
        sub = self.font_sm.render(sub_txt, True, (145, 165, 195))
        self.screen.blit(sub, (panel.right - sub.get_width() - 16, 616))

        if count_f == 0:
            done_msg = self.font_title.render(
                "✓ 0 FAILING TEST CASES — 100% VERIFIED SORTING PROGRAM!",
                True,
                (80, 235, 145),
            )
            self.screen.blit(done_msg, (28, 680))
            self.screen.set_clip(None)
            return

        N = self.array_len
        cols = 8
        chip_w = (panel.width - 48) // cols
        chip_h = 24
        start_x = panel.x + 24
        start_y = panel.y + 44

        for idx, case_id in enumerate(failing_indices):
            r = idx // cols
            c = idx % cols
            if start_y + (r + 1) * (chip_h + 5) > panel.bottom - 6:
                break
            rx = start_x + c * chip_w
            ry = start_y + r * (chip_h + 5)
            rect = pygame.Rect(rx, ry, chip_w - 8, chip_h)
            pygame.draw.rect(self.screen, (42, 30, 38), rect, border_radius=4)
            pygame.draw.rect(
                self.screen, (140, 75, 85), rect, width=1, border_radius=4
            )

            bits_list = [(int(case_id) >> b) & 1 for b in range(N)]
            bit_str = "[" + ",".join(str(b) for b in bits_list) + "]"
            surf = self.font_xs.render(bit_str, True, (245, 195, 200))
            self.screen.blit(
                surf,
                (
                    rect.centerx - surf.get_width() // 2,
                    rect.centery - surf.get_height() // 2,
                ),
            )

        self.screen.set_clip(None)

    def run(self) -> None:
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            hovered_action: Optional[int] = None
            for a_idx, rect in enumerate(self.action_rects):
                if rect.collidepoint(mouse_pos):
                    hovered_action = a_idx
                    break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        self.start_mcts_step()
                    elif event.key == pygame.K_a:
                        self.auto_play = not self.auto_play
                    elif event.key == pygame.K_b:
                        self.step_bubble_sort()
                    elif event.key == pygame.K_u:
                        self.undo_step()
                    elif event.key == pygame.K_r:
                        self.reset_episode()
                    elif event.key == pygame.K_4:
                        self.load_problem_size(4)
                    elif event.key == pygame.K_5:
                        self.load_problem_size(5)
                    elif event.key == pygame.K_6:
                        self.load_problem_size(6)
                    elif event.key == pygame.K_UP:
                        self.num_simulations = min(800, self.num_simulations + 40)
                    elif event.key == pygame.K_DOWN:
                        self.num_simulations = max(20, self.num_simulations - 40)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if (
                        hovered_action is not None
                        and hovered_action in self.env.legal_actions()
                        and self.search_gen is None
                    ):
                        self.apply_user_or_ai_action(hovered_action)

            self.update_mcts_animation()
            self.draw(hovered_action)
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    viz = SortingProgramVisualizer(array_len=5, num_simulations=120)
    viz.run()
