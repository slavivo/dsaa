import logging
from dataclasses import dataclass, field

from simple_parsing.helpers import Serializable

logger = logging.getLogger("data")

@dataclass
class AugArgs(Serializable):
    # === Waveform-level augs ===
    noise_dir: str = "noise"
    noise_prob: float = 0.0
    noise_min_snr_db: float = 10.0
    noise_max_snr_db: float = 50.0

    gain_prob: float = 0.0
    gain_min_db: float = -24.0
    gain_max_db: float = 15.0

    speed_prob: float = 0.0
    speed_min: float = 0.85
    speed_max: float = 1.15

    pitch_prob: float = 0.0
    pitch_min_cent: float = -200.0
    pitch_max_cent: float = 200.0

    echo_prob: float = 0.0
    echo_min_delay_ms: int = 100
    echo_max_delay_ms: int = 500
    echo_min_factor: float = 0.0
    echo_max_factor: float = 0.2

    # === Code-level augs ===
    temporal_shift_prob: float = 0.0
    temporal_shift_min: float = -0.56
    temporal_shift_max: float = 0.56
    temporal_shift_hz: float = 12.5
    temporal_shift_fill_token: int = -1

    text_mask_prob: float = 0.0
    text_mask_mask_prob: float = 0.3
    text_mask_token_id: int = -1

    audio_mask_prob: float = 0.0
    audio_mask_mask_prob: float = 0.3
    audio_mask_token_id: int = -1

@dataclass()
class DataArgs(Serializable):
    """
     Arguments for data loading. Train and eval data should be jsonl files
    with  "path" and "duration" fields for each audio .wav file.
    """

    train_data: list[str] | str = field(default_factory=list)
    merge_group: list[str] = field(default_factory=list)
    shuffle: bool = False
    eval_loss_data: str = ""
    mmlu_data: str = ""
    swuggy_data: str = ""
    sblimp_data: str = ""
    ssc_data: str = ""
    cd_data: list[str] | str = field(default_factory=list)

    aug: AugArgs = field(default_factory=AugArgs)