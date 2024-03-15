import torch
from torch import nn


class BitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_type: str = "1.58bit",  # "1bit" or "1.58bit"
        bits: int = 8,
        bias: bool = False,
        flg_before_linear: bool = True,
    ):
        super().__init__(in_features, out_features, bias)

        self.weight_type = weight_type
        self.bits = bits
        self.flg_before_linear = flg_before_linear

        self.norm = nn.LayerNorm(in_features)
        self.Qb = 2 ** (bits - 1)
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        # --- 1. quantized weights
        beta = torch.abs(self.weight).mean()
        if self.weight_type == "1bit":
            # binarized quantize
            alpha = self.weight.mean()
            weight = self.ste_sign(self.weight - alpha)
        elif self.weight_type == "1.58bit":
            # absmean quantization
            weight = self.weight / (beta + self.eps)
            weight = self.ste_round(weight)
            weight = torch.clamp(weight, -1, 1)
        else:
            raise ValueError(self.weight_type)

        # --- 2. quantized inputs, absmax quantization
        gamma = torch.abs(x).max()
        if self.flg_before_linear:
            # [-Qb, Qb]
            x = x * self.Qb / gamma
            x = torch.clamp(x, -self.Qb + self.eps, self.Qb - self.eps)
        else:
            # [0, Qb]
            eta = torch.min(x)
            x = (x - eta) * self.Qb / gamma
            x = torch.clamp(x, self.eps, self.Qb - self.eps)

        # --- 3. calc
        # Addition is faster than multiplication, but there is no implementation
        x = torch.nn.functional.linear(x, weight, self.bias)

        # --- 4. dequantized inputs
        x = x * gamma * beta / self.Qb

        return x

    def ste_sign(self, x):
        # torch.sign(0) -> 0 so I made it myself
        x2 = torch.where(x > 0, 1, -1)
        return (x2 - x).detach() + x  # STE

    def ste_round(self, x):
        x2 = torch.round(x)
        return (x2 - x).detach() + x  # STE


def _check_grad():

    # gradはtorchだとよくわからなかったのでtensorflowの結果を利用

    # --- linear
    x = torch.tensor([0.9, 0, -1], requires_grad=True, dtype=torch.float32)
    target = torch.tensor([1, 3, 4], dtype=torch.float32)  # 仮の目標値を定義
    criterion = torch.nn.MSELoss()  # 何らかの損失関数を定義
    y = torch.nn.functional.linear(x, x)
    loss = criterion(y, target)
    loss.backward()
    print("linear:", x.grad, y)

    # --- sign(where)
    x = torch.tensor([0.9, 0, -1], requires_grad=True, dtype=torch.float32)
    target = torch.tensor([1, 3, 4], dtype=torch.float32)
    criterion = torch.nn.MSELoss()
    y = torch.round(x)
    # y = torch.where(x > 0, 1, -1)
    loss = criterion(y, target)
    loss.backward()
    print("sign(where):", x.grad, y)

    # --- clamp
    x = torch.tensor([0.9, 1.2, 2.5], requires_grad=True, dtype=torch.float32)
    target = torch.tensor([1, 3, 4], dtype=torch.float32)
    criterion = torch.nn.MSELoss()
    y = torch.clamp(x, -1, 1)
    loss = criterion(y, target)
    loss.backward()
    print("clamp:", x.grad, y)


def _main():
    x = torch.tensor([[1, 2], [1, 1]], dtype=torch.float32)
    m = BitLinear(2, 32)
    y = m(x)
    print(y, y.shape)


if __name__ == "__main__":
    # _check_grad()
    _main()
