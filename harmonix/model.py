from __future__ import annotations
import json
from dataclasses import dataclass, field


@dataclass
class Overtone:
    frequency_factor: float  # ratio to base frequency (e.g. 2.0 = one octave above)
    volume_factor: float     # ratio to base volume

    def to_dict(self) -> dict:
        return {
            "frequency_factor": self.frequency_factor,
            "volume_factor": self.volume_factor,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Overtone:
        return cls(
            frequency_factor=float(d["frequency_factor"]),
            volume_factor=float(d["volume_factor"]),
        )


@dataclass
class Tone:
    base_frequency: float          # Hz
    base_volume: float             # 0.0–1.0 linear
    overtones: list[Overtone] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "base_frequency": self.base_frequency,
            "base_volume": self.base_volume,
            "overtones": [o.to_dict() for o in self.overtones],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Tone:
        return cls(
            base_frequency=float(d["base_frequency"]),
            base_volume=float(d["base_volume"]),
            overtones=[Overtone.from_dict(o) for o in d.get("overtones", [])],
        )


@dataclass
class ToneDatabase:
    tones: dict[str, Tone] = field(default_factory=dict)

    def add(self, name: str, tone: Tone) -> None:
        self.tones[name] = tone

    def remove(self, name: str) -> None:
        self.tones.pop(name, None)

    def to_dict(self) -> dict:
        return {name: tone.to_dict() for name, tone in self.tones.items()}

    @classmethod
    def from_dict(cls, d: dict) -> ToneDatabase:
        return cls(tones={name: Tone.from_dict(t) for name, t in d.items()})

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> ToneDatabase:
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
