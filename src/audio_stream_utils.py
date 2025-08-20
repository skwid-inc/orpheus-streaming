"""
Audio streaming utilities for WebSocket endpoints.
Provides proper audio format handling and LiveKit compatibility.
"""

import struct
from typing import AsyncGenerator, Optional
from enum import Enum

class AudioFormat(Enum):
    """Supported audio formats for streaming."""
    PCM_S16LE = "pcm_s16le"  # 16-bit signed PCM, little-endian
    OPUS = "opus"            # Opus codec for WebRTC
    WAV = "wav"              # WAV with headers

class AudioStreamConfig:
    """Configuration for audio streaming."""
    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        bits_per_sample: int = 16,
        format: AudioFormat = AudioFormat.PCM_S16LE,
        chunk_size: int = 4096,
        include_wav_header: bool = False
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bits_per_sample = bits_per_sample
        self.format = format
        self.chunk_size = chunk_size
        self.include_wav_header = include_wav_header
        
    @property
    def bytes_per_sample(self) -> int:
        return self.bits_per_sample // 8
        
    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * self.bytes_per_sample

def create_wav_header(data_size: int, config: AudioStreamConfig) -> bytes:
    """Create WAV header for audio data."""
    byte_rate = config.sample_rate * config.channels * config.bytes_per_sample
    block_align = config.channels * config.bytes_per_sample
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',                    # ChunkID
        data_size + 36,             # ChunkSize
        b'WAVE',                    # Format
        b'fmt ',                    # Subchunk1ID
        16,                         # Subchunk1Size (PCM)
        1,                          # AudioFormat (PCM)
        config.channels,            # NumChannels
        config.sample_rate,         # SampleRate
        byte_rate,                  # ByteRate
        block_align,                # BlockAlign
        config.bits_per_sample,     # BitsPerSample
        b'data',                    # Subchunk2ID
        data_size                   # Subchunk2Size
    )
    return header

async def stream_with_format(
    audio_generator: AsyncGenerator[bytes, None],
    config: AudioStreamConfig,
    include_header: bool = False
) -> AsyncGenerator[bytes, None]:
    """
    Wrap audio stream with proper format handling.
    
    Args:
        audio_generator: Raw audio byte generator
        config: Audio stream configuration
        include_header: Whether to include format headers
        
    Yields:
        Formatted audio chunks
    """
    if include_header and config.format == AudioFormat.WAV:
        # For WAV format, we need to buffer all data first
        # or use a streaming WAV approach
        audio_data = bytearray()
        async for chunk in audio_generator:
            audio_data.extend(chunk)
            
        # Send header first
        header = create_wav_header(len(audio_data), config)
        yield header
        
        # Then send data in chunks
        for i in range(0, len(audio_data), config.chunk_size):
            yield bytes(audio_data[i:i + config.chunk_size])
    else:
        # For raw PCM or when no header needed
        async for chunk in audio_generator:
            yield chunk

class AudioBuffer:
    """
    Buffer for smooth audio streaming with backpressure handling.
    """
    def __init__(self, max_size: int = 10):
        self.buffer = []
        self.max_size = max_size
        self.total_bytes = 0
        
    def add_chunk(self, chunk: bytes) -> bool:
        """Add chunk to buffer. Returns False if buffer is full."""
        if len(self.buffer) >= self.max_size:
            return False
        self.buffer.append(chunk)
        self.total_bytes += len(chunk)
        return True
        
    def get_chunk(self) -> Optional[bytes]:
        """Get next chunk from buffer."""
        if self.buffer:
            chunk = self.buffer.pop(0)
            self.total_bytes -= len(chunk)
            return chunk
        return None
        
    @property
    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size
        
    @property
    def is_empty(self) -> bool:
        return len(self.buffer) == 0
