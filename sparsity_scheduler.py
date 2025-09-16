from typing import Dict, Sequence, Tuple, Optional, Union

import torch.nn as nn


class SparsityScheduler:
    def __init__(
        self,
        model: nn.Module,
        *,
        mode: str = "uniform",
        start: float = 0.0,
        end: float = 0.5,
        total_steps: int = 100_000,
        grid: Optional[Union[Dict[int, float], Sequence[Tuple[int, float]]]] = None,
    ) -> None:
        self.model = model
        self.mode = mode
        self.start = float(start)
        self.end = float(end)
        self.total_steps = int(total_steps)

        if mode == "grid":
            if grid is None:
                raise ValueError("grid must be provided when mode='grid'")
            if isinstance(grid, dict):
                self.grid = dict(sorted(grid.items()))  
            else:
                self.grid = dict(sorted(grid))  
        else:
            self.grid = {}

        self._last_ratio: Optional[float] = None

    def _ratio_uniform(self, step: int) -> float:
        if step >= self.total_steps:
            return self.end
        return self.start + (self.end - self.start) * step / self.total_steps

    def _ratio_grid(self, step: int) -> float:
        keys = [k for k in self.grid if k <= step]
        if not keys:
            return next(iter(self.grid.values()))
        return self.grid[max(keys)]

    def _current_ratio(self, step: int) -> float:
        if self.mode == "uniform":
            return self._ratio_uniform(step)
        if self.mode == "grid":
            return self._ratio_grid(step)
        if self.mode == "static":
            return self.start
        raise ValueError("mode must be 'uniform' or 'grid'")

    def _apply_ratio(self, ratio: float) -> None:
        for m in self.model.modules():
            setter = getattr(m, "set_sparsity_ratio", None)
            if callable(setter):
                setter(ratio)

    def __call__(self, step: int) -> float:
        ratio = float(self._current_ratio(step))
        if ratio != self._last_ratio:
            self._apply_ratio(ratio)
            self._last_ratio = ratio
        return ratio
