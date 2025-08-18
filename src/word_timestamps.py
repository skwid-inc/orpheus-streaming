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
    """Simple grapheme-to-phoneme approximation."""
    
    def get_phone_counts(self, words: List[str]) -> List[int]:
        """
        Estimate phone count per word using simple heuristics.
        Real implementation would use phonemizer or g2p-en.
        """
        counts = []
        for word in words:
            # Simple heuristic: ~1.2 phones per character
            # Short words get a minimum of 2 phones
            # Long words cap at 15 phones
            length = len(word)
            if length <= 2:
                count = 2
            elif length <= 4:
                count = 3
            elif length <= 7:
                count = int(length * 1.2)
            else:
                count = min(15, int(length * 1.1))
            counts.append(count)
        return counts


class VoiceProfile:
    """Per-voice timing parameters."""
    
    def __init__(self, voice_id: str):
        self.voice_id = voice_id
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
        """Load voice profile from JSON if exists."""
        profile_path = Path(f"voice_profiles/{self.voice_id}.json")
        if profile_path.exists():
            try:
                with open(profile_path) as f:
                    data = json.load(f)
                    self.ms_per_phone = data.get("ms_per_phone", self.ms_per_phone)
                    self.pause_priors_ms.update(data.get("pause_priors_ms", {}))
                    self.min_word_ms = data.get("min_word_ms", self.min_word_ms)
                    self.speed_factors.update(data.get("speed_factors", {}))
                    self.guard_ms = data.get("guard_ms", self.guard_ms)
                    self.rescale_interval_ms = data.get("rescale_interval_ms", self.rescale_interval_ms)
                logger.info(f"Loaded voice profile for {self.voice_id}")
            except Exception as e:
                logger.warning(f"Failed to load voice profile: {e}")


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
        Rescale unfinalized words to match audio playback time.
        Returns True if timeline was updated.
        """
        guard = self.voice_profile.guard_ms / 1000.0
        n = len(self.words)
        
        # Find new finalized index (words that are definitely done playing)
        new_finalized = self.finalized_index
        while new_finalized + 1 < n and self.base_end[new_finalized + 1] <= max(0, audio_seconds - guard):
            new_finalized += 1
        
        # If everything is finalized, nothing to rescale
        if new_finalized >= n - 1:
            self.finalized_index = n - 1
            return False
        
        # Calculate scale factor for open segment
        pred_final_end = self.base_end[new_finalized] if new_finalized >= 0 else 0.0
        pred_open_end = self.base_end[-1]  # End of last word
        
        if pred_open_end - pred_final_end < 0.001:  # Avoid division by zero
            return False
        
        scale = (audio_seconds - pred_final_end) / (pred_open_end - pred_final_end)
        scale = max(0.6, min(1.5, scale))  # Clamp scale factor
        
        # Update unfinalized word timings
        updated = False
        for i in range(new_finalized + 1, n):
            new_start = pred_final_end + (self.base_start[i] - pred_final_end) * scale
            new_end = pred_final_end + (self.base_end[i] - pred_final_end) * scale
            
            # Only update if changed significantly
            if abs(new_start - self.start[i]) > 0.01 or abs(new_end - self.end[i]) > 0.01:
                self.start[i] = new_start
                self.end[i] = new_end
                updated = True
        
        self.finalized_index = new_finalized
        return updated
    
    def get_timeline_event(self) -> Dict:
        """Get current timeline as event dict."""
        return {
            "type": "TIMELINE_UPDATE",
            "words": self.words,
            "start": self.start,
            "end": self.end,
            "start_sample": [int(s * SR) for s in self.start],
            "end_sample": [int(e * SR) for e in self.end],
            "finalized_until_index": self.finalized_index,
            "source": "interpolated"
        }


class WordTimestampGenerator:
    """Main class for generating word timestamps during streaming."""
    
    def __init__(self):
        self.normalizer = TextNormalizer()
        self.g2p = SimpleG2P()
        self.voice_profiles = {}
    
    def get_voice_profile(self, voice_id: str) -> VoiceProfile:
        """Get or create voice profile."""
        if voice_id not in self.voice_profiles:
            self.voice_profiles[voice_id] = VoiceProfile(voice_id)
        return self.voice_profiles[voice_id]
    
    def create_timeline(self, text: str, voice_id: str, 
                       speaking_rate: str = "normal") -> WordTimeline:
        """Create a word timeline for the given text."""
        # Normalize and tokenize
        words, punct_classes = self.normalizer.normalize_and_tokenize(text)
        
        if not words:
            return None
        
        # Get phone counts
        phone_counts = self.g2p.get_phone_counts(words)
        
        # Get voice profile
        voice_profile = self.get_voice_profile(voice_id)
        
        # Create timeline
        return WordTimeline(words, punct_classes, phone_counts, 
                          voice_profile, speaking_rate)