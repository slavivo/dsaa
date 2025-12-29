import numpy as np
import random
import os
from torchaudio import load as audio_load
import torchaudio
import torchaudio.transforms as T
import torch
from .args import DataArgs
from typing import Literal
                
class BaseCodeAugmenter:
    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, codes: torch.tensor) -> torch.tensor:
        if random.random() < self.prob:
            return self.apply(codes)
        return codes
    
    def apply(self, codes: torch.tensor) -> torch.tensor:
        raise NotImplementedError
    
class TemporalShifter(BaseCodeAugmenter):
    def __init__(self, prob: float, min_shift: float = -0.56, max_shift: float = 0.56, 
                 hz: float = 12.5, fill_token: torch.Tensor = torch.tensor([-1])):
        super().__init__(prob)
        ms = 1000 / hz
        self.min_shift = int(min_shift * 1000 / ms)
        self.max_shift = int(max_shift * 1000 / ms)
        self.hz = hz
        self.ms = ms
        self.fill_token = fill_token
    
    def apply(self, codes: torch.Tensor) -> torch.Tensor:
        shift_tokens = random.randint(self.min_shift, self.max_shift)
        
        if codes.dim() == 3:  # (B, 17, T)
            codes_shifted = codes.clone()
            
            if shift_tokens > 0:
                # Shift and pad audio streams (1-16) forward relative to text
                pad_token = self.fill_token.to(codes.device)
                codes_shifted[:, 1:, shift_tokens:] = codes[:, 1:, :-shift_tokens]
                codes_shifted[:, 1:, :shift_tokens] = pad_token
            elif shift_tokens < 0:
                # Shift and pad text stream forward relative to audio
                pad_token = self.fill_token.to(codes.device)
                abs_shift = abs(shift_tokens)                
                codes_shifted[:, 0, abs_shift:] = codes[:, 0, :-abs_shift]
                codes_shifted[:, 0, :abs_shift] = pad_token
            
            return codes_shifted
        
        elif codes.dim() == 2:  # (17, T)
            codes_shifted = codes.clone()
            
            if shift_tokens > 0:
                # Shift and pad audio streams (1-16) forward relative to text
                codes_shifted[1:, shift_tokens:] = codes[1:, :-shift_tokens]
                codes_shifted[1:, :shift_tokens] = 0  # Pad with zeros
            elif shift_tokens < 0:
                # Shift and pad text stream forward relative to audio
                abs_shift = abs(shift_tokens)
                codes_shifted[0, abs_shift:] = codes[0, :-abs_shift]
                codes_shifted[0, :abs_shift] = self.fill_token.to(codes.device)
            return codes_shifted    
        else:
            raise ValueError(f"Unexpected tensor shape: {codes.shape}")
        
class TokenMasker(BaseCodeAugmenter):
    def __init__(self, prob: float, mask_prob: float = 0.3, mask_token_id: int = -1,
                 target: Literal["text", "audio"] = "audio"):
        super().__init__(prob)
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        if target not in ("text", "audio"):
            raise ValueError(f"Invalid target={target}, must be 'text' or 'audio'")
        self.target = target
    
    def apply(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() == 3:  # (B, 17, T)
            batch_size, num_streams, seq_len = codes.shape
            mask = torch.rand(batch_size, seq_len, device=codes.device) < self.mask_prob
            mask = mask.unsqueeze(1)  # (B, 1, T)

            mask_expanded = torch.zeros(batch_size, num_streams, seq_len).bool().to(codes.device)
            
            if self.target == "text":
                mask_expanded[:, 0, :] = mask.squeeze(1)
            else: # audio
                mask_expanded[:, 1:9, :] = mask
            
            codes_masked = codes.clone()
            codes_masked[mask_expanded] = self.mask_token_id
            return codes_masked
        
        elif codes.dim() == 2:  # (17, T)
            num_streams, seq_len = codes.shape
            mask = torch.rand(seq_len, device=codes.device) < self.mask_prob

            mask_expanded = torch.zeros(num_streams, seq_len).bool().to(codes.device)

            if self.target == "text":
                mask_expanded[0, :] = mask
            else: # audio
                mask_expanded[1:9, :] = mask
            
            codes_masked = codes.clone()
            codes_masked[mask_expanded] = self.mask_token_id
            return codes_masked
        
        else:
            raise ValueError(f"Unexpected tensor shape: {codes.shape}")
        
class BaseWaveAugmenter:
    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, sample_wav: np.ndarray, sr: int) -> np.ndarray:
        if random.random() < self.prob:
            return self.apply(sample_wav, sr)
        return sample_wav
    
    def apply(self, sample_wav: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError
    
class EchoAugmenter(BaseWaveAugmenter):
    def __init__(self, prob: float, min_delay_ms: int = 100, max_delay_ms: int = 500, 
                 min_factor: float = 0.0, max_factor: float = 0.2):
        super().__init__(prob)
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.min_factor = min_factor
        self.max_factor = max_factor

    def apply(self, sample_wav: np.ndarray, sr: int) -> np.ndarray:
        moshi = sample_wav[0]
        user = sample_wav[1]

        factor = random.uniform(self.min_factor, self.max_factor)
        delay_ms = random.uniform(self.min_delay_ms, self.max_delay_ms)
        delay_samples = int(sr * delay_ms / 1000.0)

        delayed = np.zeros_like(moshi)
        if delay_samples < len(moshi):
            delayed[delay_samples:] = moshi[:-delay_samples]

        augmented_user = user + factor * delayed

        augmented = np.stack([moshi, augmented_user], axis=0).astype(sample_wav.dtype)
        return augmented

class NoiseAugmenter(BaseWaveAugmenter):
    def __init__(self, prob: float, path: str, max_snr_db: float = 50.0, min_snr_db: float = 10.0):
        super().__init__(prob)
        if not os.path.exists(path):
            raise NotADirectoryError(f"Specified noise directory doesn't exist: {path}")
        
        self.noise_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith('.wav')
        ]

        self.max_snr_db = max_snr_db
        self.min_snr_db = min_snr_db

        self.resamplers = {}

    def apply(self, sample_wav: np.ndarray, sr: int) -> np.ndarray:
        moshi = sample_wav[0]
        user = sample_wav[1]

        selected_noise = random.choice(self.noise_files)
        noise, noise_sr = audio_load(selected_noise)

        if sr != noise_sr:
            resampler = self.resamplers.get(f"{noise_sr}_to_{sr}")
            if not resampler:
                resampler = T.Resample(
                    orig_freq=noise_sr,
                    new_freq=sr
                )
                self.resamplers[f"{noise_sr}_to_{sr}"] = resampler
            noise = resampler(noise)

        noise = noise.squeeze().numpy()

        target_len = user.shape[0]
        if noise.shape[0] < target_len:
            repeats = (target_len // noise.shape[0]) + 1
            noise = np.tile(noise, repeats)[:target_len]
        else:
            start = random.randint(0, noise.shape[0] - target_len)
            noise = noise[start:start + target_len]

        sig_power = np.mean(sample_wav.astype(np.float32) ** 2)
        noise_power = np.mean(noise ** 2)
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        desired_noise_power = sig_power / (10 ** (snr_db / 10))
        scaling = np.sqrt(desired_noise_power / (noise_power + 1e-12))
        augmented_user = user + noise * scaling

        return np.stack([moshi, augmented_user], axis=0).astype(sample_wav.dtype)

class GainAugmenter(BaseWaveAugmenter):
    def __init__(self, prob: float, min_db: float = -24., max_db: float = 15.):
        super().__init__(prob)

        self.min_db = min_db
        self.max_db = max_db

    def apply(self, sample_wav: np.ndarray, sr: int) -> np.ndarray:
        moshi = sample_wav[0]
        user = sample_wav[1]

        gain_db = random.uniform(self.min_db, self.max_db)
        gain_linear = 10 ** (gain_db / 20)
        augmented_user = user * gain_linear

        return np.stack([moshi, augmented_user], axis=0).astype(sample_wav.dtype)
    
class SpeedAugmenter(BaseWaveAugmenter):
    def __init__(self, prob: float, min_speed: float = 0.85, max_speed: float = 1.15):
        super().__init__(prob)

        self.min_speed = min_speed
        self.max_speed = max_speed

    def apply(self, sample_wav: np.ndarray, sr: int) -> np.ndarray: 
        target_speed = random.uniform(self.min_speed, self.max_speed)
        prev_dtype = sample_wav.dtype
        audio_tensor = torch.tensor(sample_wav, dtype=torch.float32)
        
        n_fft = 1024
        hop_length = 256
        window = torch.hann_window(n_fft, device=audio_tensor.device)
        time_stretch = T.TimeStretch(hop_length=hop_length, n_freq=(n_fft // 2 + 1))
        
        augmented_channels = []
        for ch in range(audio_tensor.shape[0]):
            # --- Step 1: STFT ---
            spec = torch.stft(audio_tensor[ch], n_fft=n_fft, window=window, 
                              hop_length=hop_length, return_complex=True)
            # --- Step 2: Time stretch ---
            stretched = time_stretch(spec, target_speed)
            # --- Step 3: Griffin-Lim reconstruction ---
            wav = torch.istft( stretched, n_fft=n_fft, window=window, hop_length=hop_length, 
                              length=int(audio_tensor.shape[1] / target_speed))
            augmented_channels.append(wav)
        
        augmented_audio = torch.stack(augmented_channels, dim=0)
        return augmented_audio.squeeze(0).detach().numpy().astype(prev_dtype)
    
class PitchAugmenter(BaseWaveAugmenter):
    def __init__(self, prob: float, min_cent: float = -200., max_cent: float = 200.):
        super().__init__(prob)

        self.min_cent = min_cent
        self.max_cent = max_cent

    def apply(self, sample_wav: np.ndarray, sr: int) -> np.ndarray:
        moshi = sample_wav[0]
        user = sample_wav[1]

        target_pitch = random.uniform(self.min_cent, self.max_cent)
        transform = T.PitchShift(sample_rate=sr, n_steps=target_pitch/100)        
        augmented_user = transform(torch.tensor(user, dtype=torch.float32)).detach().numpy()

        return np.stack([moshi, augmented_user.astype(sample_wav.dtype)], axis=0)

class WaveAugmenter:
    def __init__(self, args: DataArgs):
        self.augmenters = []

        aug = args.aug

        if aug.noise_prob > 0 and aug.noise_dir:
            self.augmenters.append(
                NoiseAugmenter(
                    prob=aug.noise_prob,
                    path=aug.noise_dir,
                    min_snr_db=aug.noise_min_snr_db,
                    max_snr_db=aug.noise_max_snr_db
                )
            )

        if aug.gain_prob > 0:
            self.augmenters.append(
                GainAugmenter(
                    prob=aug.gain_prob,
                    min_db=aug.gain_min_db,
                    max_db=aug.gain_max_db
                )
            )

        if aug.speed_prob > 0:
            self.augmenters.append(
                SpeedAugmenter(
                    prob=aug.speed_prob,
                    min_speed=aug.speed_min,
                    max_speed=aug.speed_max
                )
            )

        if aug.pitch_prob > 0:
            self.augmenters.append(
                PitchAugmenter(
                    prob=aug.pitch_prob,
                    min_cent=aug.pitch_min_cent,
                    max_cent=aug.pitch_max_cent
                )
            )

        if aug.echo_prob > 0:
            self.augmenters.append(
                EchoAugmenter(
                    prob=aug.echo_prob,
                    min_delay_ms=aug.echo_min_delay_ms,
                    max_delay_ms=aug.echo_max_delay_ms,
                    min_factor=aug.echo_min_factor,
                    max_factor=aug.echo_max_factor,
                )
            )

    def __call__(self, sample_wav: np.ndarray, sr: int) -> np.ndarray:
        if sample_wav.ndim != 2 or sample_wav.shape[0] != 2:
            raise ValueError(
                f"WaveAugmenter expects stereo input (2, T), got shape {sample_wav.shape}"
            )
        for aug in self.augmenters:
            sample_wav = aug(sample_wav, sr)
        return sample_wav
    
class CodeAugmenter:
    def __init__(self, args: DataArgs):
        self.augmenters = []

        aug = args.aug

        if aug.temporal_shift_prob > 0:
            self.augmenters.append(
                TemporalShifter(
                    prob=aug.temporal_shift_prob,
                    min_shift=aug.temporal_shift_min,
                    max_shift=aug.temporal_shift_max,
                    hz=aug.temporal_shift_hz,
                    fill_token=torch.tensor([aug.temporal_shift_fill_token])
                )
            )

        if aug.text_mask_prob > 0:
            self.augmenters.append(
                TokenMasker(
                    prob=aug.text_mask_prob,
                    mask_prob=aug.text_mask_mask_prob,
                    mask_token_id=aug.text_mask_token_id,
                    target="text"
                )
            )

        if aug.audio_mask_prob > 0:
            self.augmenters.append(
                TokenMasker(
                    prob=aug.audio_mask_prob,
                    mask_prob=aug.audio_mask_mask_prob,
                    mask_token_id=aug.audio_mask_token_id,
                    target="audio"
                )
            )

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        for aug in self.augmenters:
            codes = aug(codes)
        return codes