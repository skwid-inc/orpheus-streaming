"""
Minimal word timestamp generation for Orpheus streaming TTS.
Server-side interpolation with live rescaling.
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional, Literal
from pathlib import Path
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Constants
SR = 24000  # Sample rate
BYTES_PER_SAMPLE = 2  # 16-bit audio
CHANNELS = 1

class TextNormalizer:
    """Minimal text normalizer - expands numbers and handles punctuation."""
    
    def __init__(self):
        self.punct_classes = {
            '.': 'period',
            '!': 'exclaim', 
            '?': 'question',
            ',': 'comma',
            ';': 'semicolon',
            ':': 'colon'
        }
    
    def normalize_and_tokenize(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Normalize text and return (words, punctuation_classes).
        """
        # Simple tokenization by spaces
        text = text.strip()
        tokens = text.split()
        
        words = []
        punct_classes = []
        
        for token in tokens:
            if not token:
                continue
                
            # Check for trailing punctuation
            punct_class = 'none'
            word = token
            
            if token and token[-1] in self.punct_classes:
                punct_class = self.punct_classes[token[-1]]
                word = token[:-1]
            
            if word:  # Only add non-empty words
                words.append(word)
                punct_classes.append(punct_class)
        
        return words, punct_classes


class SimpleG2P:
    """Grapheme-to-phoneme using g2p-en."""
    
    def __init__(self):
        from g2p_en import G2p
        self.g2p = G2p()
        logger.info("Using g2p-en for phoneme counts")
    
    def get_phone_counts(self, words: List[str]) -> List[int]:
        """
        Get phone count per word using g2p-en.
        """
        counts = []
        for word in words:
            counts.append(self._get_cached_phoneme_count(word))
        return counts
    
    @lru_cache(maxsize=10000)
    def _get_cached_phoneme_count(self, word: str) -> int:
        """Cache g2p-en phoneme counts to avoid repeated computation."""
        phonemes = self.g2p(word.lower())
        # Filter out spaces and count actual phonemes
        phoneme_count = len([p for p in phonemes if p != ' '])
        # Minimum of 2 phonemes (even for single letters)
        return max(2, phoneme_count)


class VoiceProfile:
    """Timing parameters."""
    
    def __init__(self):
        self.ms_per_phone = 78.0
        self.pause_priors_ms = {
            "none": 0,
            "comma": 160, 
            "period": 320,
            "question": 360,
            "exclaim": 320,
            "colon": 200,
            "semicolon": 140
        }
        self.min_word_ms = 100
        self.speed_factors = {
            "slow": 1.15,
            "normal": 1.0,
            "fast": 0.85
        }
        self.guard_ms = 60
        self.rescale_interval_ms = 300
        
        # Try to load from file if exists
        self.load_from_file()
    
    def load_from_file(self):
        """Load default timing parameters."""
        # Using default values - no file loading needed


class WordTimeline:
    """Manages word timing predictions and rescaling."""
    
    def __init__(self, words: List[str], punct_classes: List[str], 
                 phone_counts: List[int], voice_profile: VoiceProfile,
                 speaking_rate: str = "normal"):
        self.words = words
        self.punct_classes = punct_classes
        self.phone_counts = phone_counts
        self.voice_profile = voice_profile
        self.speaking_rate = speaking_rate
        
        # Identify sentence end indices (., ?, ! end a sentence)
        self.sentence_end_indices = []
        for i, p in enumerate(self.punct_classes):
            if p in {"period", "question", "exclaim"}:
                self.sentence_end_indices.append(i)
        # Ensure the last word closes a segment
        if not self.sentence_end_indices or self.sentence_end_indices[-1] != len(self.words) - 1:
            self.sentence_end_indices.append(len(self.words) - 1)
        
        # Build initial predictions
        self.base_start, self.base_end = self._build_initial_timeline()
        
        # Current (possibly rescaled) timings
        self.start = self.base_start.copy()
        self.end = self.base_end.copy()
        
        # Track what's been finalized
        self.finalized_index = -1
        
    def _build_initial_timeline(self) -> Tuple[List[float], List[float]]:
        """Build initial predicted timeline."""
        speed = self.voice_profile.speed_factors.get(self.speaking_rate, 1.0)
        mpp = self.voice_profile.ms_per_phone
        min_ms = self.voice_profile.min_word_ms
        pauses = self.voice_profile.pause_priors_ms
        
        start_times = []
        end_times = []
        t = 0.0
        
        for word, phones, punct in zip(self.words, self.phone_counts, self.punct_classes):
            # Calculate word duration
            dur_ms = max(min_ms, mpp * phones) * speed
            
            # Set start and end
            s = t
            e = t + dur_ms / 1000.0
            start_times.append(s)
            end_times.append(e)
            
            # Add pause after word if punctuation
            pause_ms = pauses.get(punct, 0)
            t = e + pause_ms / 1000.0
        
        return start_times, end_times
    
    def rescale_to_audio_time(self, audio_seconds: float) -> bool:
        """
        Freeze words whose *scaled* end is behind audio, then rubber-band only the
        current sentence (between finalized+1 and its sentence end) to match audio.
        Returns True if any timings changed.
        """
        guard = self.voice_profile.guard_ms / 1000.0
        n = len(self.words)
        updated = False

        # 1) Finalize against *scaled* end times (not base)
        new_finalized = self.finalized_index
        while new_finalized + 1 < n and self.end[new_finalized + 1] <= max(0.0, audio_seconds - guard):
            new_finalized += 1
        if new_finalized != self.finalized_index:
            self.finalized_index = new_finalized
            updated = True

        # 2) If everything is finalized, nothing to scale
        if self.finalized_index >= n - 1:
            return updated

        # 3) Determine current sentence open segment [lo .. hi]
        lo = self.finalized_index + 1
        # Find first sentence end >= lo
        hi = next((idx for idx in self.sentence_end_indices if idx >= lo), n - 1)

        # 4) Compute scale using *base* anchors for the open segment
        pred_final_end = self.base_end[self.finalized_index] if self.finalized_index >= 0 else 0.0
        pred_open_end = self.base_end[hi]
        denom = max(1e-3, pred_open_end - pred_final_end)
        target = max(pred_final_end, audio_seconds)  # Don't shrink behind the anchor
        scale = (target - pred_final_end) / denom
        scale = max(0.6, min(1.5, scale))  # Clamp for stability

        # 5) Apply scaling to open segment only; enforce monotonicity & min duration
        MIN_DUR = 0.02  # 20ms minimum duration
        for i in range(lo, hi + 1):
            ns = pred_final_end + (self.base_start[i] - pred_final_end) * scale
            ne = pred_final_end + (self.base_end[i] - pred_final_end) * scale
            # Push forward to avoid overlap with finalized region
            if i > 0 and ns < self.end[i - 1]:
                ns = self.end[i - 1]
            if ne <= ns:
                ne = ns + MIN_DUR
            if abs(ns - self.start[i]) > 1e-3 or abs(ne - self.end[i]) > 1e-3:
                self.start[i] = ns
                self.end[i] = ne
                updated = True

        # 6) (Optional) Shift subsequent sentences by the same delta to preserve relative spacing
        delta = (self.end[hi] - self.base_end[hi]) if hi < n else 0.0
        if abs(delta) > 1e-3 and hi + 1 < n:
            for i in range(hi + 1, n):
                ns = self.base_start[i] + delta
                ne = self.base_end[i] + delta
                # Enforce monotonicity
                if ns < self.end[i - 1]:
                    ns = self.end[i - 1]
                if ne <= ns:
                    ne = ns + MIN_DUR
                if abs(ns - self.start[i]) > 1e-3 or abs(ne - self.end[i]) > 1e-3:
                    self.start[i] = ns
                    self.end[i] = ne
                    updated = True

        return updated
    
    def get_timeline_event(self) -> Dict:
        """Get current timeline as event dict."""
        return {
            "type": "TIMELINE_UPDATE",
            "words": self.words,  # Original text, not expanded
            "start": self.start,
            "end": self.end,
            "start_sample": [int(round(s * SR)) for s in self.start],
            "end_sample": [int(round(e * SR)) for e in self.end],
            "finalized_until_index": self.finalized_index,
            "sample_rate": SR,
            "source": "interpolated"
        }


class WordTimestampGenerator:
    """Main class for generating word timestamps during streaming."""
    
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.g2p = SimpleG2P()
        self.voice_profile = VoiceProfile()
    
    def create_timeline(self, text: str, 
                       speaking_rate: str = "normal") -> WordTimeline:
        """Create a word timeline for the given text."""
        # Normalize and tokenize
        words, punct_classes = self.normalizer.normalize_and_tokenize(text)
        
        if not words:
            return None
        
        # Get phone counts
        phone_counts = self.g2p.get_phone_counts(words)
        
        # Create timeline
        return WordTimeline(words, punct_classes, phone_counts, 
                          self.voice_profile, speaking_rate)