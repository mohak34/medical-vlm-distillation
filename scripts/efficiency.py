"""Parameters, fp32 size and latency of the teacher image encoder vs the ResNet-18 student.
The text side is cached, so only image encoders are compared. Writes results/efficiency.csv."""

import time

import pandas as pd
import torch
from torch import nn

from discipulus.data import ROOT
from discipulus.student import make_student
from discipulus.teacher import load_teacher


@torch.no_grad()
def latency_ms(model: nn.Module, device: str, batch: int, reps: int) -> float:
    model = model.to(device).eval()
    x = torch.randn(batch, 3, 224, 224, device=device)
    for _ in range(5):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


def main() -> None:
    teacher = load_teacher(device="cpu").visual
    models = {"Teacher image encoder (ViT-B/32)": teacher, "Student (ResNet-18 + 512-d proj)": make_student(512)}
    rows = []
    for name, m in models.items():
        n = sum(p.numel() for p in m.parameters())
        rows.append({
            "model": name,
            "params (M)": f"{n / 1e6:.1f}",
            "fp32 size (MB)": f"{n * 4 / 2**20:.0f}",
            "GPU ms/img, batch 1": f"{latency_ms(m, 'cuda', 1, 200):.2f}",
            "GPU ms/img, batch 64": f"{latency_ms(m, 'cuda', 64, 20) / 64:.3f}",
            "CPU ms/img, batch 1": f"{latency_ms(m, 'cpu', 1, 30):.1f}",
        })  # fmt: skip
    t = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    t.to_csv(ROOT / "results" / "efficiency.csv", index=False)
    print(f"torch threads {torch.get_num_threads()}, {torch.cuda.get_device_name()}, fp32\n")
    print(t.to_markdown(index=False))


if __name__ == "__main__":
    main()
