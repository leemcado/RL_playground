"""pygame 관전 화면 (play env).

- 왼쪽: 전체 맵 (1 게임유닛 = 8×8 도트 → 100×100 맵이 800×800 px)
- 오른쪽 위: 포커스 세포가 받는 6채널 이미지 관측 (채널별 색으로 합쳐 표시)
- 오른쪽 아래: 포커스 세포의 자기 상태, 스텝, 점수판

렌더러는 WorldState 배열만 읽으므로 엔진이 JAX 로 바뀌어도 그대로 쓸 수 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from cell_arena.core import physics
from cell_arena.core.api import IMAGE_CHANNELS, SELF_FEATURES
from cell_arena.core.config import BLACK_HOLE, FOOD, WHITE_HOLE, ArenaConfig
from cell_arena.core.observation import self_state, semantic_image
from cell_arena.core.state import WorldState

BACKGROUND = (22, 24, 34)
FOOD_COLOR = (150, 150, 158)
BLACK_FILL = (12, 6, 20)
BLACK_RIM = (130, 60, 200)
WHITE_FILL = (255, 248, 225)
WHITE_RIM = (255, 205, 110)
CELL_COLORS = (
    (80, 220, 160),
    (240, 110, 90),
    (90, 160, 250),
    (240, 200, 70),
    (210, 130, 245),
    (250, 150, 200),
    (120, 230, 240),
    (170, 210, 90),
)
# 이미지 관측 채널별 표시 색 (IMAGE_CHANNELS 순서)
CHANNEL_COLORS = ((150, 150, 158), (110, 190, 250), (150, 70, 230), (255, 215, 120), (240, 90, 80), (80, 220, 160))
assert len(CHANNEL_COLORS) == len(IMAGE_CHANNELS)

GRID = (32, 35, 48)
VISION = (90, 100, 140)
PANEL_BG = (16, 17, 24)
PANEL_BORDER = (70, 76, 100)
TEXT = (220, 222, 230)
DIM = (130, 135, 150)
BAR = (110, 170, 240)
VIEW_RESOLUTION = 64


@dataclass
class HudInfo:
    """패널 정보.

    Attributes:
        step: 경과 스텝
        max_steps: 시간 제한 (None 이면 무한)
        focus: 패널에 보여줄 세포 인덱스
        names: 에이전트 이름 목록
        status: 경기 종료 메시지, 진행 중이면 ""
        action_bits: 사람 조작 시 [W S A D SPC] 입력
        colors: 세포별 표시 색 — 에이전트 파일의 color 속성 (None 이면 기본 팔레트)
    """

    step: int
    max_steps: int | None
    focus: int
    names: list[str]
    status: str
    action_bits: np.ndarray | None = None
    colors: list | None = None


def _rgb(color: tuple[int, int, int] | list[int] | str) -> tuple[int, int, int]:
    """(R, G, B) 또는 '#RRGGBB' → (R, G, B), 0~255 로 자른다."""
    if isinstance(color, str):
        hexa = color.lstrip("#")
        return int(hexa[0:2], 16), int(hexa[2:4], 16), int(hexa[4:6], 16)
    r, g, b = (int(max(0, min(255, c))) for c in color)
    return r, g, b


def _composite(image: np.ndarray) -> np.ndarray:
    """6채널 0/1 이미지 (6, R, R) → 표시용 RGB (R, R, 3). 뒤 채널이 위에 그려진다 (자기 세포가 맨 위)."""
    rgb = np.empty((*image.shape[1:], 3), dtype=np.uint8)
    rgb[:] = BACKGROUND
    for channel, color in zip(image, CHANNEL_COLORS):
        rgb[channel > 0] = color
    return rgb


class ArenaViewer:
    """전체 맵 + 관측 패널 창.

    Args:
        cfg: 게임 규칙
        scale: 게임유닛당 픽셀 수
        obs_scale: 관측 이미지 확대 배율
    """

    def __init__(self, cfg: ArenaConfig, scale: int = 8, obs_scale: int = 5) -> None:
        pygame.init()
        self.cfg = cfg
        self.scale = scale
        self.map_px = int(round(cfg.map_size * scale))
        self.obs_px = VIEW_RESOLUTION * obs_scale
        self.pad = 20
        self.panel_w = self.obs_px + 2 * self.pad
        self.screen = pygame.display.set_mode((self.map_px + self.panel_w, self.map_px))
        pygame.display.set_caption("Cell Arena — play env (NumPy engine)")
        pygame.key.stop_text_input()  # 한글 IME 가 WASD 입력을 가로채지 않게

        mono = "menlo,sfnsmono,dejavusansmono,consolas,monospace"
        self.font = pygame.font.SysFont(mono, 14)
        self.small = pygame.font.SysFont(mono, 12)
        self.big = pygame.font.SysFont(mono, 30, bold=True)
        self._background = self._make_background()

    def draw(self, state: WorldState, hud: HudInfo) -> None:
        """한 프레임 그리기."""
        palette = self._palette(hud, state.cells.size.shape[0])
        self.screen.blit(self._background, (0, 0))
        self._draw_objects(state, palette)
        self._draw_cells(state, hud.focus, palette)
        if hud.status:
            self._draw_overlay(hud.status)
        self._draw_panel(state, hud, palette)
        pygame.display.flip()

    @staticmethod
    def _palette(hud: HudInfo, n: int) -> list[tuple[int, int, int]]:
        """세포별 표시 색: 에이전트가 정한 색, 없으면 기본 팔레트."""
        colors = list(hud.colors or [])
        return [
            _rgb(colors[i]) if i < len(colors) and colors[i] is not None else CELL_COLORS[i % len(CELL_COLORS)]
            for i in range(n)
        ]

    # ------------------------------------------------------------------ 맵

    def _make_background(self) -> pygame.Surface:
        surf = pygame.Surface((self.map_px, self.map_px))
        surf.fill(BACKGROUND)
        for p in range(0, self.map_px, 10 * self.scale):  # 10 게임유닛 격자
            pygame.draw.line(surf, GRID, (p, 0), (p, self.map_px))
            pygame.draw.line(surf, GRID, (0, p), (self.map_px, p))
        return surf

    def _wrapped(self, pos: np.ndarray, extent: float) -> list[tuple[float, float]]:
        """토러스 경계에 걸친 객체를 반대편에도 그리기 위한 픽셀 좌표 목록."""
        w = self.map_px
        x, y = pos[0] * self.scale, pos[1] * self.scale
        xs = [x] + ([x + w] if x - extent < 0 else []) + ([x - w] if x + extent > w else [])
        ys = [y] + ([y + w] if y - extent < 0 else []) + ([y - w] if y + extent > w else [])
        return [(xx, yy) for xx in xs for yy in ys]

    def _draw_objects(self, state: WorldState, palette: list[tuple[int, int, int]]) -> None:
        obj = state.objects
        radii = physics.radius(obj.size, self.cfg) * self.scale
        spin = state.step * 0.12
        for i in np.flatnonzero(obj.alive):
            t, r = obj.type[i], max(float(radii[i]), 1.0)
            if t == WHITE_HOLE:
                for x, y in self._wrapped(obj.pos[i], r):
                    pygame.draw.circle(self.screen, WHITE_RIM, (x, y), r)
                    pygame.draw.circle(self.screen, WHITE_FILL, (x, y), r * 0.7)
            elif t == BLACK_HOLE:
                for x, y in self._wrapped(obj.pos[i], r):
                    self._draw_black_hole(x, y, r, spin)
            else:
                color = FOOD_COLOR if t == FOOD else palette[obj.owner[i] % len(palette)]
                for x, y in self._wrapped(obj.pos[i], r):
                    pygame.draw.circle(self.screen, color, (x, y), r)

    def _draw_black_hole(self, x: float, y: float, r: float, spin: float) -> None:
        pygame.draw.circle(self.screen, BLACK_RIM, (x, y), r)
        pygame.draw.circle(self.screen, BLACK_FILL, (x, y), r - 2)
        rect = pygame.Rect(0, 0, int(r * 1.3), int(r * 1.3))
        rect.center = (int(x), int(y))
        for k in range(3):  # 회전하는 소용돌이 팔
            a = spin + k * 2 * math.pi / 3
            pygame.draw.arc(self.screen, BLACK_RIM, rect, a, a + 1.6, 1)

    def _draw_cells(self, state: WorldState, focus: int, palette: list[tuple[int, int, int]]) -> None:
        cells = state.cells
        w = self.map_px
        if cells.alive[focus]:  # 포커스 세포의 시야 정사각형 (토러스 wrap)
            half = float(physics.vision_radius(cells.size[focus], self.cfg)) * self.scale
            cx, cy = cells.pos[focus] * self.scale
            for ox in (-w, 0, w):
                for oy in (-w, 0, w):
                    rect = pygame.Rect(0, 0, int(2 * half), int(2 * half))
                    rect.center = (int(cx + ox), int(cy + oy))
                    pygame.draw.rect(self.screen, VISION, rect, 1)

        for i in np.argsort(cells.size):  # 작은 세포부터 그려 큰 세포가 위에
            if not cells.alive[i]:
                continue
            r = float(physics.radius(cells.size[i], self.cfg)) * self.scale
            color = palette[i]
            outline = (255, 255, 255) if cells.dashing[i] else tuple(c // 2 for c in color)
            vx, vy = cells.vel[i] * self.scale * 3
            for x, y in self._wrapped(cells.pos[i], r):
                pygame.draw.circle(self.screen, color, (x, y), r)
                pygame.draw.circle(self.screen, outline, (x, y), r, width=2)
                pygame.draw.line(self.screen, outline, (x, y), (x + vx, y + vy), 2)

    def _draw_overlay(self, status: str) -> None:
        shade = pygame.Surface((self.map_px, self.map_px), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        self.screen.blit(shade, (0, 0))
        title = self.big.render(status, True, TEXT)
        hint = self.font.render("press R to restart", True, DIM)
        c = self.map_px / 2
        self.screen.blit(title, title.get_rect(center=(c, c - 16)))
        self.screen.blit(hint, hint.get_rect(center=(c, c + 22)))

    # ------------------------------------------------------------------ 패널

    def _text(self, s: str, x: float, y: float, color: tuple[int, int, int] = TEXT,
              font: pygame.font.Font | None = None) -> None:
        self.screen.blit((font or self.font).render(s, True, color), (x, y))

    def _draw_panel(self, state: WorldState, hud: HudInfo, palette: list[tuple[int, int, int]]) -> None:
        x0, pad, h_px = self.map_px, self.pad, self.map_px
        left = x0 + pad
        focus = hud.focus
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, self.panel_w, h_px))
        pygame.draw.line(self.screen, PANEL_BORDER, (x0, 0), (x0, h_px), 2)

        # 위: 포커스 세포가 받는 이미지 관측
        h = float(physics.vision_radius(state.cells.size[focus], self.cfg))
        y = pad
        dead = "" if state.cells.alive[focus] else "  (eliminated)"
        self._text(f"VIEW  {hud.names[focus]}{dead}", left, y)
        y += 18
        self._text(f"image ({len(IMAGE_CHANNELS)},{VIEW_RESOLUTION},{VIEW_RESOLUTION}) 0/1   view {2 * h:.1f} x {2 * h:.1f} u",
                   left, y, DIM, self.small)
        y += 18
        x = left
        for name, color in zip(IMAGE_CHANNELS, CHANNEL_COLORS):
            name = name.removesuffix("_hole").removeprefix("other_")  # food cell_food black white cell self
            width = 12 + self.small.size(name)[0]
            if x + width > left + self.obs_px:  # 패널 폭을 넘으면 다음 줄
                x, y = left, y + 16
            pygame.draw.rect(self.screen, color, (x, y + 3, 9, 9))
            self._text(name, x + 12, y, DIM, self.small)
            x += width + 12
        y += 20
        view = _composite(semantic_image(state, focus, self.cfg, VIEW_RESOLUTION))
        img = pygame.surfarray.make_surface(np.ascontiguousarray(view.swapaxes(0, 1)))
        self.screen.blit(pygame.transform.scale(img, (self.obs_px, self.obs_px)), (left, y))
        pygame.draw.rect(self.screen, PANEL_BORDER, (left - 1, y - 1, self.obs_px + 2, self.obs_px + 2), 1)
        y += self.obs_px + 22

        # 아래: 공통 자기 상태
        self._text("SELF STATE  (5,)", left, y)
        y += 26
        bar_x, bar_w = left + 196, self.obs_px - 196
        # 막대 표시 범위 (음수가 될 수 있으면 0 기준 양방향 막대)
        L = self.cfg.map_size
        ranges = ((0.0, self.cfg.win_size), (0.0, L), (0.0, L), (-3.0, 3.0), (-3.0, 3.0))
        for label, value, (lo, hi) in zip(SELF_FEATURES, self_state(state.cells, focus), ranges):
            self._text(label, left, y, DIM)
            self._text(f"{value:9.2f}", left + 110, y)
            pygame.draw.rect(self.screen, PANEL_BORDER, pygame.Rect(bar_x, y + 3, bar_w, 10), 1)
            zero = bar_x + bar_w * (0.0 - lo) / (hi - lo)
            tip = bar_x + bar_w * (min(max(float(value), lo), hi) - lo) / (hi - lo)
            pygame.draw.rect(self.screen, BAR, pygame.Rect(min(zero, tip), y + 4, abs(tip - zero), 8))
            if lo < 0:
                pygame.draw.line(self.screen, DIM, (zero, y + 1), (zero, y + 15))
            y += 24

        # 스텝·입력
        y += 12
        limit = f" / {hud.max_steps}" if hud.max_steps is not None else "   (no time limit)"
        self._text(f"step {hud.step}{limit}", left, y)
        y += 19
        if hud.action_bits is not None:
            bits = " ".join(str(int(b)) for b in hud.action_bits)
            self._text(f"you   [W S A D SPC] = {bits}", left, y)
            y += 19

        # 점수판
        y += 4
        cells = state.cells
        for i in np.argsort(-np.where(cells.alive, cells.size, -1.0))[:8]:
            pygame.draw.rect(self.screen, palette[i], (left, y + 3, 9, 9))
            mark = ">" if i == focus else " "
            size = f"{cells.size[i]:7.1f}" if cells.alive[i] else "   dead"
            self._text(f"{mark} {hud.names[i][:18]:<18} {size}", left + 14, y, TEXT if cells.alive[i] else DIM,
                       self.small)
            y += 15

        keys = "WASD move  SPACE dash  " if hud.action_bits is not None else ""
        self._text(f"{keys}TAB focus  R reset  ESC quit", left, h_px - 46, DIM, self.small)
        self._text("play env: NumPy engine  |  training: JAX engine", left, h_px - 26, (200, 150, 90), self.small)
