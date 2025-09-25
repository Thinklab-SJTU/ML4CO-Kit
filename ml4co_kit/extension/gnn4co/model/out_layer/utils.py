from torch import nn, Tensor


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)
