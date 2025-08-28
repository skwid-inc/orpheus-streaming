import time
import uuid
from pathlib import Path
from typing import Iterator

import pysbd
import torch
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer

# Import the working decoder_v2 and TTS modules via absolute imports
from model.decoder_v2 import tokens_decoder
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


# Using tokens_decoder from decoder_v2 instead
async def tokens_decoder_baseten_wrapper(token_gen: Iterator, request_id: str = "") -> Iterator[bytes]:
    """Wrapper to call decoder_v2's tokens_decoder with Baseten's signature."""
    # decoder_v2's tokens_decoder accepts optional request_id and start_time parameters
    async for audio_chunk in tokens_decoder(token_gen, request_id=request_id):
        yield audio_chunk


# Commented out - using decoder_v2's implementation
# The rest of the local decoder implementation is replaced by decoder_v2


class BasetenEngineAdapter:
    """Adapter to make Baseten's engine compatible with TTSWithTimestamps"""
    
    def __init__(self, baseten_engine, model_instance):
        self.engine = baseten_engine
        self.model = model_instance
    
    async def generate_speech_async(self, prompt):
        """Generate speech using Baseten's engine and decoder_v2"""
        # Format the prompt using Model's format_prompt method
        formatted_prompt = self.model.format_prompt(prompt)
        
        # Create the input for Baseten's engine
        inp = {
            "request_id": str(uuid.uuid4()),
            "prompt": formatted_prompt,
            "max_tokens": 1200,  # Using our fixed values
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "end_id": 128258,
        }
        
        # Get token generator from Baseten's engine
        tokgen = await self.engine.predict(inp)
        if isinstance(tokgen, StreamingResponse):
            tokgen = tokgen.body_iterator
        
        # Use decoder_v2 to convert tokens to audio
        async for audio_chunk in tokens_decoder(tokgen):
            yield audio_chunk


class Model:
    def __init__(self, trt_llm, **kwargs) -> None:
        self._secrets = kwargs["secrets"]
        self._engine = trt_llm["engine"]
        self._data_dir = kwargs["data_dir"]
        self._model = None
        self._tokenizer = None
        self.websocket_connections: dict[str, dict] = {}
        self.start_id = [128259]
        self.end_ids = [128009, 128260]  # Fixed: removed extra tokens 128261, 128257
        self.text_splitter = pysbd.Segmenter(language="en", clean=False)
        
        # We'll initialize these after loading when we have the tokenizer
        self.orpheus_engine = None
        self.tts_handler = None

    def load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            Path(self._data_dir) / "tokenization"
        )
        self.start_tokenized = (
            self._tokenizer.decode(self.start_id) + self._tokenizer.bos_token
        )
        self.end_tokenized = self._tokenizer.decode(self.end_ids)

        self.use_fast_fmt = self._format_prompt_fast(
            "hello world"
        ) == self._format_prompt_slow("hello world")
        
        # Initialize the engine adapter and TTS handler after tokenizer is loaded
        self.orpheus_engine = BasetenEngineAdapter(self._engine, self)
        self.tts_handler = TTSWithTimestamps(self.orpheus_engine)

    def _format_prompt_slow(self, prompt):
        adapted_prompt = prompt
        input_ids = self._tokenizer.encode(
            adapted_prompt,
        )
        full_ids = self.start_id + input_ids + self.end_ids
        return self._tokenizer.decode(full_ids, skip_special_tokens=False)

    def _format_prompt_fast(self, prompt):
        token_stream = self.start_tokenized
        token_stream += prompt
        token_stream += self.end_tokenized
        return token_stream

    def format_prompt(self, prompt: str):
        """Format the prompt for the model."""
        if self.use_fast_fmt:
            return self._format_prompt_fast(prompt)
        else:
            print("Warn: Using slow format")
            return self._format_prompt_slow(prompt)

    async def websocket(self, ws: WebSocket):
        """Delegate to TTSWithTimestamps handler for proper websocket handling."""
        await self.tts_handler.handle_websocket(ws)
        
    async def websocket_old(self, ws: WebSocket):
        # satisfy Truss’s metrics/cancellation wrapper
        async def _never_disconnected():
            return False

        ws.is_disconnected = _never_disconnected

        sid = str(uuid.uuid4())
        print(f"[ws:{sid}] entered at {time.time()}")

        # 1) receive metadata
        params = await ws.receive_json()
        print(f"[ws:{sid}] metadata: {params!r}")
        max_tokens = params.get("max_tokens", 1200)  # Fixed: was 6144
        temperature = 0.1
        top_p = params.get("top_p", 0.95)  # Fixed: was 0.8
        rep_pen = params.get("repetition_penalty", 1.1)  # Fixed: was 1.3
        buf_sz = int(params.get("buffer_size", 10))
        print(f"buffer_size={buf_sz}")

        # initialize per-sid state
        self.websocket_connections[sid] = {
            "text_buffer": [],  # this is your cache
            # you could add more, e.g. "audio_buffer": []
        }

        async def flush(final=False):
            buf = self.websocket_connections[sid]["text_buffer"]
            if not buf:
                return
            full_text = " ".join(buf)
            sentences = self.text_splitter.segment(full_text)
            if len(sentences) > 1:  # flush all complete sentences
                complete_sents = sentences[:-1]
                prompt = " ".join(complete_sents)
                words_consumed = sum(len(s.split()) for s in complete_sents)
                del buf[:words_consumed]
            elif len(buf) >= buf_sz:
                # forcefully clear the buffer based on buffer size
                chunk = buf[:buf_sz]
                prompt = " ".join(chunk)
                del buf[:buf_sz]
            elif final:  # Clear the remaining buffer
                prompt = " ".join(buf)
                buf.clear()
            else:
                return

            print(f"Flushing prompt: {prompt}")

            inp = {
                "request_id": sid,
                "prompt": self.format_prompt(prompt),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": rep_pen,
                "end_id": 128258,
            }

            tokgen = await self._engine.predict(inp, ws)
            if isinstance(tokgen, StreamingResponse):
                tokgen = tokgen.body_iterator

            sent = 0
            async for audio in tokens_decoder_baseten_wrapper(tokgen, sid):
                sent += len(audio)
                # print(f"[ws:{sid}] sending {len(audio)} bytes (total {sent})")
                await ws.send_bytes(audio)

            if final:
                print(f"[ws:{sid}] final flush complete - closing")
                await ws.close()

        try:
            # 2) receive loop
            while True:
                text = await ws.receive_text()
                # print(f"[ws:{sid}] got text: {text!r}")

                if text == "__END__":
                    print(f"[ws:{sid}] END sentinel received")
                    await flush(final=True)
                    break

                # append to your cached buffer
                self.websocket_connections[sid]["text_buffer"].extend(
                    text.strip().split()
                )

                # flush in chunks
                await flush()

        except WebSocketDisconnect:
            print(f"[ws:{sid}] disconnected unexpectedly - final flush")
            await flush(final=True)

        finally:
            print(f"[ws:{sid}] handler exit, clearing cache")
            # optionally inspect or persist your cache here:
            # cached_words = self.websocket_connections[sid]["text_buffer"]
            del self.websocket_connections[sid]