import os
from typing import List, Optional, Tuple
import numpy as np
import pygame
import torch

from cube_env import ACTION_NAMES, FACE_NAMES, PocketCubeEnv
from mcts import MinMaxStats, Node, SinglePlayerMCTS
from model import SortingProgramNet


# Standard Rubik's Cube high-contrast sticker palette:
# 0: U = White, 1: D = Yellow, 2: F = Green, 3: R = Red, 4: L = Orange, 5: B = Blue
STICKER_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (242, 245, 250),  # 0: U (White)
    (250, 215, 45),   # 1: D (Yellow)
    (45, 205, 95),    # 2: F (Green)
    (235, 65, 60),    # 3: R (Red)
    (250, 140, 35),   # 4: L (Orange)
    (55, 130, 245),   # 5: B (Blue)
)


def _shade_color(
    rgb: Tuple[int, int, int], factor: float
) -> Tuple[int, int, int]:
    return (
        max(0, min(255, int(rgb[0] * factor))),
        max(0, min(255, int(rgb[1] * factor))),
        max(0, min(255, int(rgb[2] * factor))),
    )


class PocketCubeVisualizer:
    def __init__(self, initial_scramble: int = 5, num_simulations: int = 50):
        pygame.init()
        self.width = 1180
        self.height = 820
        pygame.display.set_caption(
            "2x2x2 Pocket Rubik's Cube — Backward Curriculum vs. Naive AlphaZero"
        )
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("menlo", 16, bold=True)
        self.font_md = pygame.font.SysFont("menlo", 14, bold=True)
        self.font_sm = pygame.font.SysFont("menlo", 12)
        self.font_xs = pygame.font.SysFont("menlo", 11)

        self.max_steps = 14
        self.num_simulations = num_simulations
        self.scramble_depth = initial_scramble
        self.auto_play = False
        self.action_rects: List[pygame.Rect] = []

        self.env = PocketCubeEnv(max_steps=self.max_steps)
        self.saved_scramble_stickers: np.ndarray = self.env.stickers.copy()
        self.saved_scramble_moves: List[int] = []
        self.history_stack: List[PocketCubeEnv] = []

        self.model_mode = "curriculum"  # "curriculum" or "naive"
        self.models = {}
        self.ckpt_status = {}
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for mode in ("curriculum", "naive"):
            net = SortingProgramNet(
                num_test_cases=144,
                num_actions=6,
                hidden_dim=128,
                num_blocks=3,
            )
            ckpt_path = os.path.join(base_dir, f"cube_{mode}.pt")
            loaded = False
            if os.path.exists(ckpt_path):
                state_dict = torch.load(
                    ckpt_path, map_location="cpu", weights_only=True
                )
                net.load_state_dict(state_dict)
                loaded = True
            net.eval()
            self.models[mode] = net
            self.ckpt_status[mode] = loaded

        self.mcts = SinglePlayerMCTS(
            self.models[self.model_mode], c_puct=1.5, discount=1.0
        )
        self.new_scramble(self.scramble_depth)

    def toggle_model(self) -> None:
        self.model_mode = (
            "naive" if self.model_mode == "curriculum" else "curriculum"
        )
        self.mcts = SinglePlayerMCTS(
            self.models[self.model_mode], c_puct=1.5, discount=1.0
        )
        self.auto_play = False
        self.search_gen = None
        self.sim_progress = 0
        self._refresh_static_evaluation()

    def new_scramble(self, depth: int) -> None:
        self.scramble_depth = depth
        self.env.reset(scramble_depth=depth)
        self.saved_scramble_stickers = self.env.stickers.copy()
        self.saved_scramble_moves = list(self.env.scramble_moves)
        self.history_stack = []
        self.auto_play = False
        self.search_gen = None
        self.sim_progress = 0
        self._refresh_static_evaluation()

    def replay_same_scramble(self) -> None:
        """Resets back to the exact same initial scramble so Curriculum and Naive
        models can be compared head-to-head on identical starting cubes.
        """
        self.env.stickers = self.saved_scramble_stickers.copy()
        self.env.scramble_moves = list(self.saved_scramble_moves)
        self.env.steps_taken = 0
        self.env.last_action = None
        self.env.move_history = []
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
        A = self.env.num_actions
        self.legal_mask = np.zeros(A, dtype=bool)
        legal = self.env.legal_actions()
        self.legal_mask[legal] = True

        active_net = self.models[self.model_mode]
        obs = torch.from_numpy(self.env.get_observation()).unsqueeze(0)
        logits, val = active_net(obs)
        logits_1d = logits.squeeze(0)

        if len(legal) > 0:
            masked_logits = torch.full_like(logits_1d, -float("inf"))
            masked_logits[legal] = logits_1d[legal]
            self.priors = torch.softmax(masked_logits, dim=-1).numpy()
        else:
            self.priors = np.zeros(A, dtype=np.float32)

        self.net_v = 0.0 if self.env.is_solved() else float(val.item())
        self.mcts_v = self.net_v
        self.visit_probs = np.zeros(A, dtype=np.float32)
        self.visit_counts = np.zeros(A, dtype=np.int32)
        self.q_values = np.full(A, np.nan, dtype=np.float32)
        self.q_norms = np.zeros(A, dtype=np.float32)
        self.q_min = float("inf")
        self.q_max = -float("inf")

    def start_mcts_Animation(self) -> None:
        if self.env.is_terminal():
            self.auto_play = False
            return
        self.search_gen = self.mcts.search_step_by_step(
            self.env, self.num_simulations, add_noise=False
        )
        self.sim_progress = 0

    def update_mcts_animation(self, sims_per_frame: int = 10) -> None:
        if self.search_gen is None:
            return
        finished = False
        root: Optional[Node] = None
        min_max_stats: Optional[MinMaxStats] = None

        for _ in range(sims_per_frame):
            try:
                sim_idx, root, root_net_v, min_max_stats = next(self.search_gen)
                self.sim_progress = sim_idx
                self.net_v = root_net_v
            except StopIteration:
                finished = True
                break

        if root is not None and min_max_stats is not None:
            self.mcts_v = root.value
            self.q_min = min_max_stats.minimum
            self.q_max = min_max_stats.maximum
            A = self.env.num_actions
            visits = np.zeros(A, dtype=np.float32)
            for a, child in root.children.items():
                visits[a] = float(child.visit_count)
                self.visit_counts[a] = child.visit_count
                if child.visit_count > 0:
                    q = root.child_q_value(child, self.mcts.discount)
                    self.q_values[a] = q
                    self.q_norms[a] = min_max_stats.normalize(q)
            total_v = float(np.sum(visits))
            if total_v > 0:
                self.visit_probs = visits / total_v

        if finished:
            self.search_gen = None
            best_a = int(np.argmax(self.visit_counts))
            self.apply_action(best_a)

    def apply_action(self, action: int) -> None:
        if self.env.is_terminal():
            return
        if action not in self.env.legal_actions():
            return
        self.history_stack.append(self.env.clone())
        self.env.step(action)
        self.search_gen = None
        self.sim_progress = 0
        self._refresh_static_evaluation()
        if self.env.is_terminal():
            self.auto_play = False

    def _draw_header(self) -> None:
        panel = pygame.Rect(16, 12, self.width - 32, 68)
        pygame.draw.rect(self.screen, (24, 32, 44), panel, border_radius=8)
        pygame.draw.rect(self.screen, (52, 70, 95), panel, width=1, border_radius=8)
        self.screen.set_clip(panel)

        mode_label = (
            "BACKWARD CURRICULUM (K=1..7)"
            if self.model_mode == "curriculum"
            else "NAIVE FORWARD (Fixed K=7)"
        )
        mode_col = (
            (85, 235, 150)
            if self.model_mode == "curriculum"
            else (255, 125, 110)
        )
        title_str = (
            f"2x2x2 Pocket Cube  |  Scramble: k={self.scramble_depth}  |  "
            f"Steps: {self.env.steps_taken}/{self.max_steps}  |  Active Model: "
        )
        t_surf = self.font_title.render(title_str, True, (235, 242, 252))
        self.screen.blit(t_surf, (28, 20))
        m_surf = self.font_title.render(mode_label, True, mode_col)
        self.screen.blit(m_surf, (28 + t_surf.get_width(), 20))

        controls_str = (
            "[1-7] Scramble k   [T] Replay Same Cube   [M] Toggle Curriculum/Naive   "
            "[Space] MCTS Step   [A] Auto   [Click] Manual Move   [Z] Undo"
        )
        c_surf = self.font_xs.render(controls_str, True, (165, 182, 205))
        self.screen.blit(c_surf, (28, 50))
        self.screen.set_clip(None)

    def _draw_cube_panel(self) -> None:
        panel = pygame.Rect(16, 90, 740, 440)
        pygame.draw.rect(self.screen, (22, 28, 38), panel, border_radius=8)
        pygame.draw.rect(self.screen, (45, 58, 78), panel, width=1, border_radius=8)
        self.screen.set_clip(panel)

        hdr = self.font_md.render(
            "3D Isometric Cube (U, F, R) & 24-Sticker Net",
            True,
            (215, 228, 248),
        )
        self.screen.blit(hdr, (28, 102))

        if self.env.is_solved():
            badge_str = f"SOLVED in {self.env.steps_taken} moves!"
            badge_col = (80, 240, 150)
        elif self.env.is_terminal():
            badge_str = f"TIMED OUT at {self.env.steps_taken} moves"
            badge_col = (250, 105, 105)
        else:
            badge_str = "Anchor: DBL (6,18,23)"
            badge_col = (145, 165, 195)
        b_surf = self.font_sm.render(badge_str, True, badge_col)
        self.screen.blit(b_surf, (panel.right - b_surf.get_width() - 16, 104))

        # --- 1. Draw 3D Isometric Cube on the Left Half (cx=195, cy=315) ---
        cx, cy = 195.0, 315.0
        R = 54.0
        vx, vy = 0.8660254 * R, -0.5 * R
        vR = np.array([vx, vy])     # Up-right vector
        vL = np.array([-vx, vy])    # Up-left vector
        vD = np.array([0.0, R])     # Straight-down vector
        C = np.array([cx, cy])      # Center UFR vertex

        def draw_iso_quad(
            origin: np.ndarray,
            du: np.ndarray,
            dv: np.ndarray,
            sticker_idx: int,
            shade: float,
        ) -> None:
            p0 = origin
            p1 = origin + du
            p2 = origin + du + dv
            p3 = origin + dv
            pts = [
                (int(p0[0]), int(p0[1])),
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (int(p3[0]), int(p3[1])),
            ]
            color_id = int(self.env.stickers[sticker_idx])
            fill_col = _shade_color(STICKER_COLORS[color_id], shade)
            pygame.draw.polygon(self.screen, fill_col, pts)
            pygame.draw.polygon(self.screen, (18, 22, 30), pts, width=3)

            center_pt = (p0 + p2) * 0.5
            idx_lbl = self.font_xs.render(
                str(sticker_idx), True, (25, 30, 40)
            )
            self.screen.blit(
                idx_lbl,
                (
                    int(center_pt[0]) - idx_lbl.get_width() // 2,
                    int(center_pt[1]) - idx_lbl.get_height() // 2,
                ),
            )

        # Top Face U (stickers 0, 1, 2, 3)
        u_top = C + 2.0 * vL + 2.0 * vR
        for r in range(2):
            for c in range(2):
                s_idx = 2 * r + c
                orig = u_top - float(r) * vR - float(c) * vL
                draw_iso_quad(orig, -vR, -vL, s_idx, shade=1.02)

        # Front Face F (stickers 8, 9, 10, 11)
        f_top_left = C + 2.0 * vL
        for r in range(2):
            for c in range(2):
                s_idx = 8 + 2 * r + c
                orig = f_top_left - float(c) * vL + float(r) * vD
                draw_iso_quad(orig, -vL, vD, s_idx, shade=0.90)

        # Right Face R (stickers 12, 13, 14, 15)
        for r in range(2):
            for c in range(2):
                s_idx = 12 + 2 * r + c
                orig = C + float(c) * vR + float(r) * vD
                draw_iso_quad(orig, vR, vD, s_idx, shade=0.78)

        # Labels around the 3D Isometric Cube
        u_lbl = self.font_md.render("U (Up)", True, (200, 215, 240))
        f_lbl = self.font_md.render("F (Front)", True, (200, 215, 240))
        r_lbl = self.font_md.render("R (Right)", True, (200, 215, 240))
        self.screen.blit(u_lbl, (int(cx) - u_lbl.get_width() // 2, int(cy - 2.35 * R)))
        self.screen.blit(f_lbl, (int(cx - 1.75 * R), int(cy + 2.05 * R)))
        self.screen.blit(r_lbl, (int(cx + 0.85 * R), int(cy + 2.05 * R)))

        # --- 2. Draw 2D Unfolded Cross Net on the Right Half ---
        face_coords = {
            0: (1, 0),  # U
            4: (0, 1),  # L
            2: (1, 1),  # F
            3: (2, 1),  # R
            5: (3, 1),  # B
            1: (1, 2),  # D
        }
        net_x0 = 382
        net_y0 = 154
        tile = 34
        face_span = tile * 2 + 18

        for face_id, (fc, fr) in face_coords.items():
            fx = net_x0 + fc * face_span
            fy = net_y0 + fr * face_span
            bg_rect = pygame.Rect(fx - 3, fy - 3, tile * 2 + 6, tile * 2 + 6)
            pygame.draw.rect(self.screen, (35, 44, 60), bg_rect, border_radius=5)

            for r in range(2):
                for c in range(2):
                    s_idx = face_id * 4 + 2 * r + c
                    sx = fx + c * tile
                    sy = fy + r * tile
                    s_rect = pygame.Rect(sx + 1, sy + 1, tile - 2, tile - 2)
                    c_id = int(self.env.stickers[s_idx])
                    pygame.draw.rect(
                        self.screen,
                        STICKER_COLORS[c_id],
                        s_rect,
                        border_radius=4,
                    )
                    if s_idx in (6, 18, 23):
                        pygame.draw.rect(
                            self.screen, (20, 25, 35), s_rect, width=2, border_radius=4
                        )
                        txt = f"{s_idx}*"
                    else:
                        txt = str(s_idx)
                    lbl = self.font_xs.render(txt, True, (25, 30, 40))
                    self.screen.blit(
                        lbl,
                        (
                            s_rect.centerx - lbl.get_width() // 2,
                            s_rect.centery - lbl.get_height() // 2,
                        ),
                    )

            f_tag = self.font_xs.render(
                FACE_NAMES[face_id], True, (175, 195, 225)
            )
            self.screen.blit(f_tag, (fx + tile - f_tag.get_width() // 2, fy - 16))

        note = self.font_xs.render(
            "* 6, 18, 23 = Fixed DBL Anchor Corner",
            True,
            (140, 158, 185),
        )
        self.screen.blit(note, (net_x0 + 86, net_y0 + 3 * face_span - 4))
        self.screen.set_clip(None)

    def _draw_telemetry_And_value(self, hovered_action: Optional[int]) -> None:
        panel = pygame.Rect(768, 90, 396, 440)
        pygame.draw.rect(self.screen, (22, 28, 38), panel, border_radius=8)
        pygame.draw.rect(self.screen, (45, 58, 78), panel, width=1, border_radius=8)
        self.screen.set_clip(panel)

        hdr = self.font_md.render(
            "MCTS Action Telemetry (6 Turns)", True, (215, 228, 248)
        )
        self.screen.blit(hdr, (panel.x + 14, panel.y + 12))

        sub = self.font_xs.render(
            "Blue: Prior P(s,a)  |  Green: Visits π(a|s)",
            True,
            (145, 165, 195),
        )
        self.screen.blit(sub, (panel.x + 14, panel.y + 32))

        A = self.env.num_actions
        self.action_rects = []
        col_w = (panel.width - 28) // A
        bar_top = panel.y + 58
        bar_bottom = panel.y + 235
        max_bar_h = bar_bottom - bar_top

        for a in range(A):
            cx = panel.x + 14 + a * col_w
            rect = pygame.Rect(cx + 2, bar_top, col_w - 4, bar_bottom - bar_top + 48)
            self.action_rects.append(rect)

            is_legal = bool(self.legal_mask[a])
            is_hovered = hovered_action == a

            bg_col = (
                (36, 48, 66)
                if is_hovered and is_legal
                else ((28, 35, 48) if is_legal else (22, 24, 30))
            )
            pygame.draw.rect(self.screen, bg_col, rect, border_radius=5)

            p_val = float(self.priors[a])
            v_val = float(self.visit_probs[a])
            bw = max(8, (rect.width - 12) // 2)

            if is_legal and p_val > 0:
                ph = int(p_val * max_bar_h)
                p_rect = pygame.Rect(rect.x + 4, bar_bottom - ph, bw, ph)
                pygame.draw.rect(self.screen, (65, 145, 245), p_rect, border_radius=3)

            if is_legal and v_val > 0:
                vh = int(v_val * max_bar_h)
                v_rect = pygame.Rect(rect.x + 6 + bw, bar_bottom - vh, bw, vh)
                pygame.draw.rect(self.screen, (65, 220, 135), v_rect, border_radius=3)

            act_lbl = self.font_md.render(
                ACTION_NAMES[a],
                True,
                (235, 242, 252) if is_legal else (105, 110, 125),
            )
            self.screen.blit(
                act_lbl,
                (rect.centerx - act_lbl.get_width() // 2, bar_bottom + 6),
            )

            if not is_legal and not self.env.is_terminal():
                tag = self.font_xs.render("UNDO", True, (215, 95, 95))
                self.screen.blit(
                    tag, (rect.centerx - tag.get_width() // 2, bar_bottom + 26)
                )
            elif self.visit_counts[a] > 0:
                qn = self.q_norms[a]
                q_str = f"Q̄:{qn:.2f}"
                ql = self.font_xs.render(q_str, True, (165, 225, 190))
                self.screen.blit(
                    ql, (rect.centerx - ql.get_width() // 2, bar_bottom + 26)
                )

        # Divider + Value & MinMaxStats readout inside bottom of right panel
        div_y = panel.y + 300
        pygame.draw.line(
            self.screen,
            (45, 58, 78),
            (panel.x + 14, div_y),
            (panel.right - 14, div_y),
            width=1,
        )

        v1 = self.font_md.render(
            f"Net V(s):  {self.net_v:6.2f} moves  (System 1)",
            True,
            (90, 175, 255),
        )
        v2 = self.font_md.render(
            f"MCTS V(s): {self.mcts_v:6.2f} moves  (System 2)",
            True,
            (75, 230, 145),
        )
        self.screen.blit(v1, (panel.x + 16, div_y + 14))
        self.screen.blit(v2, (panel.x + 16, div_y + 42))

        if self.q_max >= self.q_min:
            q_str = f"Tree MinMaxStats: Q_min={self.q_min:.2f}, Q_max={self.q_max:.2f}"
        else:
            q_str = "Tree MinMaxStats: Press [Space] to run MCTS lookahead"
        q_surf = self.font_xs.render(q_str, True, (150, 168, 195))
        self.screen.blit(q_surf, (panel.x + 16, div_y + 74))

        hint_str = "Tip: Press [M] then [T] to race Curriculum vs Naive!"
        h_surf = self.font_xs.render(hint_str, True, (235, 195, 95))
        self.screen.blit(h_surf, (panel.x + 16, div_y + 102))

        self.screen.set_clip(None)

    def _draw_move_tape_panel(self) -> None:
        panel = pygame.Rect(16, 542, self.width - 32, 264)
        pygame.draw.rect(self.screen, (22, 28, 38), panel, border_radius=8)
        pygame.draw.rect(self.screen, (45, 58, 78), panel, width=1, border_radius=8)
        self.screen.set_clip(panel)

        hdr = self.font_md.render(
            "Reverse-Scramble Path vs. Agent Solution Trajectory",
            True,
            (215, 228, 248),
        )
        self.screen.blit(hdr, (28, 554))

        # 1. Scramble Moves Row
        s_lbl = self.font_sm.render(
            f"Scramble Applied from Solved (k={len(self.saved_scramble_moves)}):",
            True,
            (170, 188, 215),
        )
        self.screen.blit(s_lbl, (28, 588))

        x = 340
        for idx, m in enumerate(self.saved_scramble_moves):
            chip = pygame.Rect(x, 582, 54, 28)
            pygame.draw.rect(self.screen, (42, 52, 70), chip, border_radius=5)
            pygame.draw.rect(self.screen, (85, 105, 140), chip, width=1, border_radius=5)
            txt = self.font_sm.render(
                f"{idx+1}:{ACTION_NAMES[m]}", True, (220, 230, 245)
            )
            self.screen.blit(
                txt,
                (
                    chip.centerx - txt.get_width() // 2,
                    chip.centery - txt.get_height() // 2,
                ),
            )
            x += 62

        # Exact Group Inverse Hint Row
        inv_moves = [ACTION_NAMES[m ^ 1] for m in reversed(self.saved_scramble_moves)]
        inv_str = " -> ".join(inv_moves) if inv_moves else "(None)"
        inv_lbl = self.font_xs.render(
            f"Exact Group Inverse (Guaranteed {len(inv_moves)}-move path back to Solved):  {inv_str}",
            True,
            (135, 195, 165),
        )
        self.screen.blit(inv_lbl, (28, 622))

        # 2. Agent's Executed Moves Row
        a_lbl = self.font_sm.render(
            f"Agent Moves ({self.model_mode.upper()} model):",
            True,
            (170, 188, 215),
        )
        self.screen.blit(a_lbl, (28, 660))

        x = 340
        if not self.env.move_history:
            empty_s = self.font_sm.render(
                "No moves yet — Press [Space] for MCTS step or [A] for Auto-Solve",
                True,
                (130, 145, 170),
            )
            self.screen.blit(empty_s, (x, 660))
        else:
            for idx, m in enumerate(self.env.move_history):
                chip = pygame.Rect(x, 654, 54, 28)
                col_bg = (35, 75, 55) if self.env.is_solved() else (38, 55, 82)
                col_bd = (80, 225, 145) if self.env.is_solved() else (85, 145, 230)
                pygame.draw.rect(self.screen, col_bg, chip, border_radius=5)
                pygame.draw.rect(self.screen, col_bd, chip, width=1, border_radius=5)
                txt = self.font_sm.render(
                    f"{idx+1}:{ACTION_NAMES[m]}", True, (235, 245, 250)
                )
                self.screen.blit(
                    txt,
                    (
                        chip.centerx - txt.get_width() // 2,
                        chip.centery - txt.get_height() // 2,
                    ),
                )
                x += 62

        # 3. Pedagogical Takeaway Banner at Bottom
        box = pygame.Rect(28, 704, self.width - 56, 88)
        pygame.draw.rect(self.screen, (18, 24, 34), box, border_radius=6)
        pygame.draw.rect(self.screen, (42, 56, 78), box, width=1, border_radius=6)

        lines = [
            "Why Backward Curricula Work on Invertible Groups (DeepCubeA / AlphaTensor):",
            "• Naive Forward Self-Play starting at k=7 faces 6 * 5^6 = 93,750 branches with 0 intermediate reward;",
            "  every episode times out (0.0% solve rate) and V(s) collapses to predicting timeout everywhere.",
            "• Reverse-Scramble Curriculum grows a calibrated value basin V(s_k) ≈ -k outward from Solved (k=1 -> 7),",
            "  so forward MCTS at depth k only needs a 1-step lookahead to fall into the depth-(k-1) basin!",
        ]
        ly = 711
        for i, line in enumerate(lines):
            col = (220, 232, 250) if i == 0 else (155, 172, 198)
            fnt = self.font_sm if i == 0 else self.font_xs
            s = fnt.render(line, True, col)
            self.screen.blit(s, (40, ly))
            ly += 16

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
                    elif event.key in (
                        pygame.K_1,
                        pygame.K_2,
                        pygame.K_3,
                        pygame.K_4,
                        pygame.K_5,
                        pygame.K_6,
                        pygame.K_7,
                    ):
                        k = event.key - pygame.K_0
                        self.new_scramble(k)
                    elif event.key == pygame.K_t:
                        self.replay_same_scramble()
                    elif event.key == pygame.K_m:
                        self.toggle_model()
                    elif event.key == pygame.K_z:
                        self.undo_step()
                    elif event.key == pygame.K_SPACE:
                        if self.search_gen is None:
                            self.start_mcts_Animation()
                    elif event.key == pygame.K_a:
                        self.auto_play = not self.auto_play
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if hovered_action is not None and self.search_gen is None:
                        self.apply_action(hovered_action)

            if self.auto_play and self.search_gen is None and not self.env.is_terminal():
                self.start_mcts_Animation()

            if self.search_gen is not None:
                self.update_mcts_animation(sims_per_frame=10)

            self.screen.fill((15, 20, 28))
            self._draw_header()
            self._draw_cube_panel()
            self._draw_telemetry_And_value(hovered_action)
            self._draw_move_tape_panel()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    vis = PocketCubeVisualizer(initial_scramble=5, num_simulations=50)
    vis.run()
