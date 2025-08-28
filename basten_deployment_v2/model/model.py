import time
import uuid
from pathlib import Path
from typing import Iterator

import pysbd
import torch
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

# Import engine and TTS handler (original working flow)
from model.trt_engine import OrpheusModelTRT
from model.tts_with_timestamps import TTSWithTimestamps

# force inference mode during the lifetime of the script
_inference_mode_raii_guard = torch._C._InferenceMode(True)
# torch.backends.cuda.matmul.allow_tf32 = True

# Removed duplicate SnacModelBatched class and model_snac instance
# These are already defined in decoder_v2.py which we're importing


# Commented out - using decoder_v2 instead
# def turn_token_into_id(token_string: int, index: int):
#     """Extract and convert the last custom token ID from a string."""
#     return token_string - 10 - ((index % 7) * 4096)


# def split_custom_tokens(s: str) -> List[int]:
#     """
#     Extracts all substrings enclosed in <custom_token_…> from the input string.
#     """
#     matches = _TOKEN_RE.findall(s)
#     return [int(match) for match in matches if match != "0"]


"""Model using original Orpheus TRT engine + decoder_v2 via TTSWithTimestamps."""


class Model:
    def __init__(self, trt_llm, **kwargs) -> None:
        self._secrets = kwargs.get("secrets", {})
        self._data_dir = kwargs.get("data_dir")
        self.engine: OrpheusModelTRT | None = None
        self.tts_handler: TTSWithTimestamps | None = None

    def load(self) -> None:
        # Initialize original TRT engine and TTS handler
        self.engine = OrpheusModelTRT()
        self.tts_handler = TTSWithTimestamps(self.engine)

    # Prompt formatting handled inside OrpheusModelTRT
    def format_prompt(self, prompt: str):
        return prompt

    async def websocket(self, ws: WebSocket):
        # Delegate entirely to the working websocket handler
        await self.tts_handler.handle_websocket(ws)