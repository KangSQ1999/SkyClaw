"""SkyClaw Visualization Utilities - Real-time matplotlib animation of HAPS coverage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class HapsVisualizer:
    """
    Real-time matplotlib visualization for SkyClaw HAPS simulation.

    Updates scatter plots and coverage circles in-place for smooth animation.
    Supports optional GIF frame capture and export.
    """

    def __init__(self, config: dict[str, Any], save_gif: bool = False, gif_path: str | None = None) -> None:
        self.config = config

        space_cfg = config["simulation"]["space"]
        self.map_width: float = float(space_cfg["width"])
        self.map_height: float = float(space_cfg["height"])

        haps_cfg = config["haps"]
        self.coverage_radius: float = float(haps_cfg["coverage_radius"])

        self.map_size: float = max(self.map_width, self.map_height)

        # GIF saving configuration
        self.save_gif: bool = save_gif
        self.gif_path: str | None = gif_path
        self.frames: list[Any] = []

        # Interactive mode setup
        plt.ion()

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # Plot element placeholders (initialized in setup_plot)
        self.user_scatter: Any = None
        self.haps_scatter: Any = None
        self.coverage_circles: list[plt.Circle] = []

        self.timestep_unit: str = config["simulation"].get("timestep_unit", "hours")

        logger.info(
            f"Visualizer initialized: {self.map_width}x{self.map_height} km, "
            f"coverage radius: {self.coverage_radius} km"
        )
        if save_gif:
            logger.info(f"GIF saving enabled, will save to: {gif_path}")

    def setup_plot(self, num_haps: int = 0) -> None:
        """Set up static plot elements: axes, grid, labels, and empty scatter containers."""
        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)

        # Visual indicator of the user distribution region
        from matplotlib.patches import Rectangle
        region = Rectangle(
            (300, 0), 600, 1000,
            linewidth=2,
            edgecolor='green',
            facecolor='green',
            alpha=0.05,
            linestyle='--',
            label='User Distribution Region'
        )
        self.ax.add_patch(region)
        self.ax.set_aspect('equal')

        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_xlabel('X (km)', fontsize=15)
        self.ax.set_ylabel('Y (km)', fontsize=15)
        self.ax.set_title('SkyClaw Simulation: Initializing...', fontsize=18, fontweight='bold')

        # Mobile users - blue dots, Stationary users - gray dots
        self.mobile_scatter = self.ax.scatter(
            [], [],
            c='blue',
            s=20,
            alpha=0.6,
            label='mobile node',
            zorder=3
        )
        self.stationary_scatter = self.ax.scatter(
            [], [],
            c='gray',
            s=16,
            alpha=0.4,
            label='fixed node',
            zorder=2
        )

        # HAPS - green triangles
        self.haps_scatter = self.ax.scatter(
            [], [],
            c='green',
            marker='^',
            s=280,
            edgecolors='green',
            linewidths=2.5,
            label='HAPS',
            zorder=4
        )

        # Pre-allocate coverage circles (invisible initially)
        for _ in range(max(num_haps, 4)):
            circle = plt.Circle(
                (0, 0),
                self.coverage_radius,
                color='orange',
                alpha=0.15,
                fill=True,
                zorder=1
            )
            self.ax.add_patch(circle)
            self.coverage_circles.append(circle)

        self.ax.legend(loc='upper left', fontsize=14)
        self.fig.tight_layout()
        logger.debug("Plot setup complete with static elements")

    def update_frame(
        self,
        user_positions: npt.NDArray[np.float64],
        haps_positions: npt.NDArray[np.float64],
        current_timestamp: int,
        movement_flags: npt.NDArray[np.int32] | None = None
    ) -> None:
        """
        Update visualization for current simulation frame.

        Efficiently updates scatter positions in-place without clearing axes.
        """
        # Separate mobile and stationary users
        if movement_flags is not None:
            mobile_mask = movement_flags == 1
            mobile_positions = user_positions[mobile_mask]
            stationary_positions = user_positions[~mobile_mask]
        else:
            mobile_positions = user_positions
            stationary_positions = np.zeros((0, 2))

        # Update scatter offsets
        self.mobile_scatter.set_offsets(mobile_positions)
        self.stationary_scatter.set_offsets(stationary_positions)
        self.haps_scatter.set_offsets(haps_positions)

        # Update coverage circles
        num_haps = len(haps_positions)
        while len(self.coverage_circles) < num_haps:
            circle = plt.Circle(
                (0, 0),
                self.coverage_radius,
                color='orange',
                alpha=0.15,
                fill=True,
                zorder=1
            )
            self.ax.add_patch(circle)
            self.coverage_circles.append(circle)

        for i, circle in enumerate(self.coverage_circles):
            if i < num_haps:
                circle.center = (haps_positions[i, 0], haps_positions[i, 1])
                circle.set_visible(True)
            else:
                circle.set_visible(False)

        # Update title with current time and user counts
        time_str = f"{current_timestamp:02d}:00"
        mobile_count = len(mobile_positions) if movement_flags is not None else len(user_positions)
        stationary_count = len(stationary_positions) if movement_flags is not None else 0
        self.ax.set_title(
            f'SkyClaw Simulation - Time: {time_str} | '
            f'Mobile: {mobile_count}, Stationary: {stationary_count}',
            fontsize=12,
            fontweight='bold'
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Capture frame for GIF if enabled
        if self.save_gif:
            from PIL import Image
            import io
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
            buf.seek(0)
            img = Image.open(buf)
            img_copy = img.copy()
            self.frames.append(img_copy)
            buf.close()
            img.close()

        plt.pause(0.1)  # Minimal pause for GUI event processing

    def keep_open(self) -> None:
        """Keep the visualization window open after simulation ends."""
        plt.ioff()
        self.ax.set_title(
            f'SkyClaw Simulation - Complete (Final State)',
            fontsize=13,
            fontweight='bold'
        )
        plt.show()
        logger.info("Visualization window closed by user")

    def save_snapshot(self, filepath: str) -> None:
        """Save current frame to file."""
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Snapshot saved to {filepath}")

    def save_gif_animation(self, output_path: str | None = None, duration: int = 500) -> str:
        """Save collected frames as GIF animation. Returns path to saved file."""
        if not self.frames:
            logger.warning("No frames collected, cannot save GIF")
            return ""

        save_path = output_path or self.gif_path
        if not save_path:
            logger.error("No output path specified for GIF")
            return ""

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        self.frames[0].save(
            save_path,
            save_all=True,
            append_images=self.frames[1:],
            duration=duration,
            loop=0,
            optimize=True
        )
        logger.info(f"GIF animation saved to {save_path} ({len(self.frames)} frames)")
        return save_path
