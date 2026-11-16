"""Policy size knobs (M3 Task 2). Defaults land the model at 10-15M parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    width: int = 256
    lstm: int = 512
    fusion_layers: int = 4
    player_layers: int = 3
    heads: int = 8
    cnn_channels: tuple = (64, 96, 128)
    win_channels: tuple = (48, 96)
    ff_mult: int = 4
    ptr_dim: int = 64  # key/query size of the pointer heads
    fine_dim: int = 8  # full-resolution minimap key size (far pointer)
    blocks_per_stage: int = 1
    mm_stem_stride: int = 4  # minimap CNN starts at 128 / 4 = 32x32 (throughput, M3 Task 6 Step 5)
    win_stem_stride: int = 2  # window CNN starts at 16x16
    amp: bool = True  # bf16 autocast for encoders on CUDA
    checkpoint_cnn: bool = True  # recompute CNN activations in backward (training memory)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cnn_channels"] = list(self.cnn_channels)
        d["win_channels"] = list(self.win_channels)
        return d

    @staticmethod
    def from_dict(d: dict | None) -> ModelConfig:
        d = dict(d or {})
        for k in ("cnn_channels", "win_channels"):
            if k in d:
                d[k] = tuple(d[k])
        known = {f for f in ModelConfig.__dataclass_fields__}
        return ModelConfig(**{k: v for k, v in d.items() if k in known})
