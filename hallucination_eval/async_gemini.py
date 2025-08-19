"""Async Gemini judge for TTS evaluation."""

import os
import json
import tempfile
import wave
from typing import Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


EVALUATION_PROMPT = """You are an expert TTS audio evaluator with deep expertise in speech synthesis quality assessment.
Your task is to provide a thorough, consistent evaluation of synthesized speech against the reference text.

Reference Text:
"{text}"

## CRITICAL LISTENING PROTOCOL
1. Listen to the ENTIRE audio at least twice
2. Focus on synthesis quality, NOT audio recording quality
3. Be consistent - use the same standards across all evaluations
4. Consider the intended use case (phone agent, customer service, etc.)

## Evaluation Instructions

Evaluate ONLY what you hear in the audio. Base your scores on these exact criteria:

### 1. hallucinations_level
- "none": Audio matches text exactly with no additions (EXPECTED: numbers/symbols may be verbalized, e.g., "$100" → "one hundred dollars", "%" → "percent", "&" → "and")
- "medium": 1-2 word insertions, minor repetitions, or slight paraphrasing beyond expected number verbalization
- "high": 3+ word additions, significant repetitions, gibberish, or wrong content beyond expected verbalizations

### 2. alnum_bad_pct
Calculate percentage of mispronounced alphanumeric content:
- Count mispronounced numbers, URLs, codes, abbreviations
- Divide by total alphanumeric items
- Return 0-100

### 3. sudden_pitch_changes
Count abrupt, unnatural pitch jumps (not emotional emphasis)

### 4. expressiveness (1-5)
**Rating Guidelines:**
- 1: Completely monotone, robotic delivery with no pitch variation
- 2: Minimal intonation, mostly flat but some attempt at variation
- 3: Basic appropriate intonation patterns, adequate for communication
- 4: Good emotional range with natural variation
- 5: Rich, nuanced expression matching professional human speech

### 5. healthy_disfluencies (1-5)  
**Rating Guidelines:**
- 1: Unnatural flow - either no pauses or awkward breaks
- 2: Some pausing but placed incorrectly
- 3: Acceptable pausing, mostly natural
- 4: Good natural rhythm with appropriate breaks
- 5: Perfect human-like pacing with natural micro-pauses

### 6. naturalness (1-5)
**Overall Human-likeness Rating:**
- 1: Obviously synthetic - immediately recognizable as TTS
- 2: Clearly synthetic but intelligible
- 3: Good quality but still detectable as TTS by careful listener
- 4: Very natural - only minor tells reveal it's synthetic
- 5: Indistinguishable from professional human recording

### 7. notes
Max 50 words explaining key issues or strengths

### 8. confidence (0.0-1.0)
Your confidence in this evaluation

## Critical Scoring Rules
- Return ONLY valid JSON
- **Be CONSISTENT** - use the same standards for every evaluation
- Score 5 only for truly exceptional, professional-grade quality
- Score 3 represents "acceptable" quality for production use
- Score 1-2 indicates significant issues that would impact usability
- Focus on synthesis quality, NOT recording quality or background noise
- If audio is corrupted/silent, set all scores to minimum values

## Scoring Calibration Guide:
- **Professional Human Recording** = 5
- **High-Quality TTS (e.g., news anchor quality)** = 4
- **Standard Commercial TTS** = 3
- **Basic/Robotic TTS** = 2
- **Poor/Unintelligible TTS** = 1

## Expected TTS Behaviors (DO NOT count as hallucinations):
- Number expansion: "$1,234" → "one thousand two hundred thirty-four dollars"
- Symbol verbalization: "%" → "percent", "&" → "and", "#" → "pound/hash"
- Phone numbers: "1-800-555-0123" → "one eight hundred five five five zero one two three"
- Decimals: "3.14" → "three point one four" or "three point fourteen"
- Abbreviations: "Dr." → "doctor", "St." → "street", "vs." → "versus"
- Time: "2:30 PM" → "two thirty PM" or "two thirty in the afternoon"
- Dates: "Jan 15" → "January fifteenth"

Return format:
{{
  "hallucinations_level": "none|medium|high",
  "alnum_bad_pct": 0-100,
  "sudden_pitch_changes": integer,
  "expressiveness": 1-5,
  "healthy_disfluencies": 1-5,
  "naturalness": 1-5,
  "notes": "string",
  "confidence": 0.0-1.0
}}"""


def calculate_wpm_from_bytes(audio_bytes: bytes, text: str) -> float:
    """Calculate WPM from audio bytes."""
    # Save to temp file to read with wave
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        with wave.open(tmp_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            # Check if frames is max int32
            if frames == 2147483647:
                # Calculate from file size
                file_size = len(audio_bytes)
                data_size = file_size - 44  # WAV header
                duration_seconds = data_size / (rate * channels * sample_width)
            else:
                duration_seconds = frames / float(rate)
            
            word_count = len(text.split())
            if duration_seconds > 0:
                wpm = (word_count / duration_seconds) * 60
            else:
                wpm = 0.0
            
            return round(wpm, 1)
    finally:
        os.unlink(tmp_path)


def evaluate_with_gemini_sync(model, audio_bytes: bytes, text: str, prompt_template: str) -> Dict[str, Any]:
    """Evaluate audio with Gemini (synchronous function)."""
    # Save audio to temp file for Gemini
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        # Upload file to Gemini
        audio_file = genai.upload_file(tmp_path)
        
        # Generate evaluation
        prompt = prompt_template.format(text=text)
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        response = model.generate_content(
            [prompt, audio_file],
            safety_settings=safety_settings
        )
        
        # Parse JSON from response
        try:
            # First try direct parsing
            result = json.loads(response.text)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown/code or bare fragments
            import re

            cleaned_text = response.text
            # Strip code fences
            if '```json' in cleaned_text:
                cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
                cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
            elif '```' in cleaned_text:
                cleaned_text = re.sub(r'```\s*', '', cleaned_text)

            # Try strict parse on cleaned text
            try:
                result = json.loads(cleaned_text.strip())
            except json.JSONDecodeError:
                # Last resort: extract best-effort JSON object
                json_match = re.search(r'\{[\s\S]*\}', cleaned_text)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except Exception:
                        result = None  # fall through to heuristic
                else:
                    result = None

                # Heuristic fallback: pull label and confidence from plain text
                if result is None:
                    level_match = re.search(r'hallucinations?_level\"?\'?\s*[:=]\s*\"?(none|medium|high)\"?', cleaned_text, re.IGNORECASE)
                    if not level_match:
                        level_match = re.search(r'\b(none|medium|high)\b', cleaned_text, re.IGNORECASE)
                    level = (level_match.group(1).lower() if level_match else 'high')

                    conf_match = re.search(r'confidence\"?\'?\s*[:=]\s*([01](?:\.\d+)?)', cleaned_text)
                    confidence = float(conf_match.group(1)) if conf_match else 0.5

                    notes_match = re.search(r'notes\"?\'?\s*[:=]\s*\"([^\"]{0,200})\"', cleaned_text)
                    notes = notes_match.group(1) if notes_match else cleaned_text[:120]

                    result = {
                        'hallucinations_level': level,
                        'confidence': confidence,
                        'notes': notes,
                        '_parser_fallback': True,
                        '_raw': cleaned_text[:500],
                    }
        
        # Calculate WPM programmatically
        wpm = calculate_wpm_from_bytes(audio_bytes, text)
        result['wpm'] = wpm
        
        return result
    finally:
        os.unlink(tmp_path)
        # Clean up uploaded file
        try:
            audio_file.delete()
        except:
            pass


