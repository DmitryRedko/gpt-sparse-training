from __future__ import annotations

from typing import Literal, Optional, List

import torch
from torch import nn

__all__ = [
    "LinearPruner",
    "replace_linears_with_pruner",
]

class LinearPruner(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity_type: Optional[str] = None,
        sparsity_ratio: float = 0.0,
        name: Optional[str] = None,

        debug_info: bool = False,
        l1_calculation: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_type = sparsity_type
        self.sparsity_ratio = float(sparsity_ratio)
        self.name = name

        self.debug_info = debug_info
        self.l1_calculation = l1_calculation

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.in_l1, self.out_l1 = None, None
        self.params_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.real_weight_sparsity, self.real_input_sparsity, self.real_output_sparsity = None, None, None

    def __repr__(self) -> str:  
        extra = (
            f", sparsity_type={self.sparsity_type}, sparsity_ratio={self.sparsity_ratio}"
            if self.sparsity_type is not None
            else ""
        )
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}{extra})"
        )

    @staticmethod
    def _compute_mask_rowwise(x: torch.Tensor, ratio: float) -> torch.Tensor:
        """0/1-маска для активаций, считается вдоль последней оси.
        Вырубает *k* наименьших |x| на каждом токене.
        """
        if ratio <= 0.0:
            return torch.ones_like(x, dtype=torch.bool)
        if ratio >= 1.0:
            return torch.zeros_like(x, dtype=torch.bool)

        *batch_dims, features = x.shape
        k = int(ratio * features)

        if k == 0:
            return torch.ones_like(x, dtype=torch.bool)
        if k >= features:
            return torch.zeros_like(x, dtype=torch.bool)

        abs_x = x.abs()
        threshold = torch.kthvalue(abs_x, k, dim=-1, keepdim=True).values
        return abs_x >= threshold

    @staticmethod
    def _compute_mask_global(w: torch.Tensor, ratio: float) -> torch.Tensor:
        """0/1-маска для весов – глобальный порог по всему тензору."""
        if ratio <= 0.0:
            return torch.ones_like(w, dtype=torch.bool)
        if ratio >= 1.0:
            return torch.zeros_like(w, dtype=torch.bool)

        flat = w.abs().flatten()
        k = int(ratio * flat.numel())
        if k == 0:
            return torch.ones_like(w, dtype=torch.bool)
        threshold = torch.kthvalue(flat, k).values
        return w.abs() >= threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        ratio = self.sparsity_ratio

        if self.sparsity_type == "masked-activations-layer":
            a_mask = self._compute_mask_rowwise(x, ratio).to(x.dtype)
            x = x * a_mask

        weight = self.weight
        if self.sparsity_type == "masked-weights-layer":
            w_mask = self._compute_mask_rowwise(weight, ratio).to(weight.dtype)
            weight = weight * w_mask

        out = torch.matmul(x, weight.t())
        if self.bias is not None:
            out = out + self.bias

        if self.l1_calculation:
            self.in_l1 = torch.nn.functional.l1_loss(x, torch.zeros_like(x), size_average=None, reduce=None, reduction='mean')
            self.out_l1 = torch.nn.functional.l1_loss(out, torch.zeros_like(out), size_average=None, reduce=None, reduction='mean')

        if self.debug_info:
            self.real_input_sparsity = (x == 0).float().mean().item()
            self.real_output_sparsity = (out == 0).float().mean().item()
            self.real_weight_sparsity = (weight == 0).float().mean().item()

        return out
    
    def get_l1_loss(self, l1_target: Literal["weight", "input", "output"] = "weight") -> torch.Tensor:
        if l1_target == "weight":
            return torch.nn.functional.l1_loss(self.weight, torch.zeros_like(self.weight), size_average=None, reduce=None, reduction='mean')
        elif l1_target == "input":
            if self.in_l1 is None:
                raise ValueError("in_l1 is not set. Run a forward pass first and check l1_calculation option.")
            return self.in_l1
        elif l1_target == "output":
            if self.out_l1 is None:
                raise ValueError("out_l1 is not set. Run a forward pass first and check l1_calculation option.")
            return self.out_l1
        else:
            raise ValueError("l1_target must be 'weight', 'input', or 'output'")

    def set_sparsity_ratio(self, sparsity_ratio: float) -> None:
        self.sparsity_ratio = float(sparsity_ratio)

    @classmethod
    def from_original(
        cls,
        orig_linear: nn.Linear,
        sparsity_type: Optional[str] = None,
        sparsity_ratio: float = 0.0,
        name: Optional[str] = None,
    ) -> LinearPruner:
        pruner = cls(
            orig_linear.in_features,
            orig_linear.out_features,
            bias=orig_linear.bias is not None,
            sparsity_type=sparsity_type,
            sparsity_ratio=sparsity_ratio,
            name=name,

            debug_info=True,
            l1_calculation=True,
        )
        pruner.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            pruner.bias.data.copy_(orig_linear.bias.data)
        return pruner

def set_sparsity_ratio(model: nn.Module, ratio: float):
    for m in model.modules():
        if isinstance(m, LinearPruner):
            m.set_sparsity_ratio(ratio)   # просто перезаписываем атрибут, его изменение не влияет на поведение оптимизатора


def replace_linears_with_pruner(
    module: nn.Module,
    sparsity_ratio: float,
    sparsity_type: str = "masked-activations-layer",
    mode: str = "all",  # "all", "exclude-first-last", or "custom"
    custom_slice: Optional[slice] = None,  # for "custom" mode
) -> List[LinearPruner]:
    linear_layers = [
        (name, layer)
        for name, layer in module.named_modules()
        if isinstance(layer, nn.Linear) or isinstance(layer, LinearPruner)
    ]

    if mode == "all":
        to_replace = set(name for name, _ in linear_layers)
    elif mode == "exclude-first-last" and len(linear_layers) > 2:
        to_replace = set(name for name, _ in linear_layers[1:-1])
    elif mode == "custom":
        if custom_slice is None:
            raise ValueError("custom_slice must be provided when mode='custom'")
        to_replace = set(name for name, _ in linear_layers[custom_slice])
    else:
        raise ValueError("mode must be 'all', 'exclude-first-last', or 'custom'")

    resulting_layers = []
    
    for name, layer in module.named_modules():
        for child_name, child in layer.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear) and full_name in to_replace:
                pruner = LinearPruner.from_original(
                    child,
                    sparsity_type=sparsity_type,
                    sparsity_ratio=sparsity_ratio,
                    name=full_name,
                ).to(child.weight.device)
                setattr(layer, child_name, pruner)
                resulting_layers.append(pruner)

            elif isinstance(child, LinearPruner):
                child.set_sparsity_ratio(sparsity_ratio)
                child.sparsity_type = sparsity_type
                resulting_layers.append(child)

    return resulting_layers
