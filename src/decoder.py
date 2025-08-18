from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import re
import logging


snac_device = "cuda" if torch.cuda.is_available() else "cpu"

_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
SNAC_MAX_BATCH = 64

if torch.cuda.is_available():
    PREPROCESS_STREAM = torch.cuda.Stream()
else:
    PREPROCESS_STREAM = None


class SnacModelBatched:
    def __init__(self):
        self.dtype_decoder = torch.float32
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(snac_device)
        self.snac_model.decoder = self.snac_model.decoder.to(self.dtype_decoder)
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._maybe_compile()

    def _maybe_compile(self):
        use_compile = True
        if not hasattr(torch, "compile"):
            return
        if not use_compile:
            return
        try:
            decoder = torch.compile(self.snac_model.decoder, dynamic=True)
            quantizer = torch.compile(self.snac_model.quantizer, dynamic=True)
            # Light warmup on a few batch sizes to specialize kernels
            for bs_size in (1, 4, min(16, SNAC_MAX_BATCH)):
                codes = [
                    torch.randint(1, 4096, (bs_size, 4), device=snac_device),
                    torch.randint(1, 4096, (bs_size, 8), device=snac_device),
                    torch.randint(1, 4096, (bs_size, 16), device=snac_device),
                ]
                with torch.inference_mode():
                    intermed = quantizer.from_codes(codes)
                    decoder(intermed.to(self.dtype_decoder))
            self.snac_model.decoder = decoder
            self.snac_model.quantizer = quantizer
        except Exception as exc:
            logging.warning(f"SNAC torch.compile skipped due to: {exc}")

    @property
    def device(self):
        return snac_device

    @property
    def dtype(self):
        return self.dtype_decoder

    # Dynamically batch multiple decode requests that arrive within a short window
    @staticmethod
    def _can_batch(all_codes):
        first_shapes = tuple(t.shape for t in all_codes[0])
        return len(all_codes) > 1 and all(tuple(t.shape for t in codes) == first_shapes for codes in all_codes)

    # Decorator imported from the `batched` package
    import batched  # local import to avoid top-level failure if not installed

    @batched.dynamically(batch_size=SNAC_MAX_BATCH, timeout_ms=15)
    def batch_snac_model(self, items: list[dict[str, list[torch.Tensor]]]) -> list[torch.Tensor]:
        with torch.inference_mode():
            if self.stream is not None:
                cm = torch.cuda.stream(self.stream)
            else:
                # Dummy context manager
                class _Noop:
                    def __enter__(self):
                        return None
                    def __exit__(self, exc_type, exc, tb):
                        return False
                cm = _Noop()

            with cm:
                all_codes = [entry["codes"] for entry in items]
                if self._can_batch(all_codes):
                    stacked_codes = [torch.cat([codes[i] for codes in all_codes], dim=0) for i in range(3)]
                    z_q = self.snac_model.quantizer.from_codes(stacked_codes)
                    audio = self.snac_model.decoder(z_q.to(self.dtype_decoder))[:, :, 2048:4096].to(torch.float32)
                    outputs = list(audio.split(1, dim=0))
                else:
                    outputs = []
                    for codes in all_codes:
                        z_q = self.snac_model.quantizer.from_codes(codes)
                        audio = self.snac_model.decoder(z_q.to(self.dtype_decoder))[:, :, 2048:4096].to(torch.float32)
                        outputs.append(audio)
            if self.stream is not None:
                self.stream.synchronize()
            return outputs


model_snac = SnacModelBatched()


def _extract_all_custom_token_values(s: str):
    return [int(m.group(1)) for m in _TOKEN_RE.finditer(s) if m.group(1) != "0"]


def turn_token_into_id(token_value: int, index: int):
    return token_value - 10 - ((index % 7) * 4096)


@torch.inference_mode()
async def convert_to_audio(frame_ids: list[int]):
    n = len(frame_ids) // 7
    if n == 0:
        return None
    arr = torch.tensor(frame_ids[: n * 7], dtype=torch.int32)
    mat = arr.view(n, 7)
    codes_0 = mat[:, 0]
    codes_1 = mat[:, [1, 4]].reshape(-1)
    codes_2 = mat[:, [2, 3, 5, 6]].reshape(-1)
    if (
        ((codes_0 < 0) | (codes_0 > 4096)).any()
        or ((codes_1 < 0) | (codes_1 > 4096)).any()
        or ((codes_2 < 0) | (codes_2 > 4096)).any()
    ):
        return None
    if PREPROCESS_STREAM is not None:
        with torch.cuda.stream(PREPROCESS_STREAM):
            codes = [
                codes_0.unsqueeze(0).to(snac_device),
                codes_1.unsqueeze(0).to(snac_device),
                codes_2.unsqueeze(0).to(snac_device),
            ]
        PREPROCESS_STREAM.synchronize()
    else:
        codes = [
            codes_0.unsqueeze(0),
            codes_1.unsqueeze(0),
            codes_2.unsqueeze(0),
        ]
        codes = [t.to(snac_device) for t in codes]
    audio_hat = await model_snac.batch_snac_model.acall({"codes": codes})
    audio_cpu = audio_hat.detach().cpu().numpy()
    audio_bytes = (audio_cpu * 32767.0).round().astype(np.int16).tobytes()
    return audio_bytes


async def tokens_decoder(token_gen):
    buffer = []
    count = 0
    first_chunk_enqueued = False
    PROCESS_EVERY = 7
    WINDOW = 28
    audio_queue = asyncio.Queue()

    async def producer():
        nonlocal buffer, count, first_chunk_enqueued
        consumed = 0
        async for token_sim in token_gen:
            toks = _extract_all_custom_token_values(token_sim)
            if len(toks) <= consumed:
                continue
            new_toks = toks[consumed:]
            consumed = len(toks)
            for raw_tok in new_toks:
                token_id = turn_token_into_id(int(raw_tok), count)
                if token_id <= 0:
                    continue
                buffer.append(token_id)
                count += 1
                if not first_chunk_enqueued and count >= 7:
                    task = asyncio.create_task(convert_to_audio(buffer[-7:]))
                    audio_queue.put_nowait(task)
                    first_chunk_enqueued = True
                elif count % PROCESS_EVERY == 0 and count >= WINDOW:
                    task = asyncio.create_task(convert_to_audio(buffer[-WINDOW:]))
                    audio_queue.put_nowait(task)
        audio_queue.put_nowait(None)

    producer_task = asyncio.create_task(producer())

    while True:
        task = await audio_queue.get()
        if task is None:
            break
        audio = await task
        if audio is not None:
            yield audio
        audio_queue.task_done()
    await producer_task


def tokens_decoder_sync(syn_token_gen):

    audio_queue = queue.Queue()

    # Convert the synchronous token generator into an async generator.
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        # tokens_decoder.tokens_decoder is assumed to be an async generator that processes tokens.
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio

    thread.join()