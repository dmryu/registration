from typing import Union
from typing_extensions import Literal
import torch
import torch.nn.functional as F
from effnet import efficientnet_b5, _initialize_weight_goog
import cv2
import numpy as np

N = None


class PCC(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        dims = list(set(range(input.dim())) - set([0]))
        std_x = input.std(dim=dims, keepdim=True)
        std_y = target.std(dim=dims, keepdim=True)

        mean_x = input.mean(dim=dims, keepdim=True)
        mean_y = target.mean(dim=dims, keepdim=True)

        vx = (input - mean_x) / std_x
        vy = (target - mean_y) / std_y

        pcc = (vx * vy).mean(dim=dims)
        if self.reduction == "mean":
            pcc = pcc.mean()
        elif self.reduction == "sum":
            pcc = pcc.sum()

        return 1 - pcc


class Registrator(object):
    def __init__(
        self,
        lr: float = 0.01,
        loss: Literal["ncc", "pcc", "mse"] = "pcc",
        in_chans: int = 3,
        num_epochs: int = 200,
        th_earlystopping: float = 1e-6,
        device: Union[int, str] = 0,
    ) -> None:
        self.device = torch.device(f"cuda:{device}")
        self.net = efficientnet_b5(num_classes=6, in_chans=in_chans).cuda(
            device=self.device
        )
        if loss == "pcc":
            self.loss_fn = PCC()
        elif loss == "ncc":
            self.loss_fn = (
                lambda m, f: 1
                - F.conv2d(m, f) / ((m**2).sum() * (f**2).sum()).sqrt()
            )
        elif loss == "mse":
            self.loss_fn = torch.nn.MSELoss()  # (m, f)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.th_earlystopping = th_earlystopping
        self.initialize()

    def initialize(self) -> None:
        for m in self.net.modules():
            _initialize_weight_goog(m)
        self.net.classifier.weight.data.zero_()
        self.net.classifier.bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def train(self, m: np.ndarray, f: np.ndarray) -> torch.Tensor:
        m = torch.from_numpy(m).permute(2, 0, 1)[N].cuda(device=self.device).float()
        f = torch.from_numpy(f).permute(2, 0, 1)[N].cuda(device=self.device).float()
        self.ncc_input = self.loss_fn(m, f).item()

        prev_loss = 0
        loss_cnt = 0

        for i in range(self.num_epochs):
            self.net.train()
            self.theta = self.net(m).view(-1, 2, 3)
            grid = F.affine_grid(self.theta, m.shape, align_corners=False)
            out = F.grid_sample(m, grid, padding_mode="zeros", align_corners=False)

            self.optimizer.zero_grad()
            loss = self.loss_fn(out, f)
            self.ncc_output = loss.item()
            loss.backward()
            self.optimizer.step()

            if abs(self.ncc_output - prev_loss) < self.th_earlystopping:
                loss_cnt += 1
            prev_loss = self.ncc_output

            if loss_cnt == 10:
                return out

        return out

    @torch.no_grad()
    def infer(self, m: np.ndarray) -> np.ndarray:
        theta = self.theta.detach().cpu()
        m_t = torch.from_numpy(m).permute(2, 0, 1)[N].float() #.cuda(device=self.device)
        grid = F.affine_grid(theta, m_t.shape, align_corners=False)
        out = F.grid_sample(m_t, grid, padding_mode="zeros", align_corners=False)
#         return out[0].cpu().permute(1, 2, 0).numpy()
        return out[0].permute(1, 2, 0).numpy()
