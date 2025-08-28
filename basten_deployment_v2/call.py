import asyncio
import os

import pyaudio
from aiohttp import ClientSession, WSMsgType
from dotenv import load_dotenv

load_dotenv()

API_KEY = "hyUQgCX6.l1DkdmprvdXOGHTuPFGXachilSKEyuqi"

MODEL_ID = "rwnd64y3"

# WS_URL = f"wss://model-{MODEL_ID}.api.baseten.co/environments/production/websocket"

WS_URL = "wss://model-rwnd64y3.api.baseten.co/deployment/w76d2d3/websocket"

MAX_TOKENS = 2000
BUFFER_SIZE = 10  # words / chunk
SAMPLE_RATE = 24000
WIDTH = pyaudio.paInt16
CHANNELS = 1


async def stream_tts(text: str):
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)

    headers = {"Authorization": f"Api-Key {API_KEY}"}
    print(f"Connecting to WebSocket: {WS_URL}")
    async with ClientSession(headers=headers) as sess:
        try:
            async with sess.ws_connect(WS_URL) as ws:
                print("✅ WS connected")

                # send metadata once
                await ws.send_json(
                    {
                        "max_tokens": MAX_TOKENS,
                        "buffer_size": BUFFER_SIZE,
                        "voice": "tara"
                    }
                )
                print("📤 metadata sent")

                # start audio receiver
                async def receiver():
                    async for msg in ws:
                        if msg.type == WSMsgType.BINARY:
                            print(f"⏯️  playing {len(msg.data)} bytes")
                            stream.write(msg.data)
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED):
                            print("🔒 server closed")
                            return

                recv = asyncio.create_task(receiver())

                # send words
                for w in text.strip().split():
                    await ws.send_str(w)
                print("📤 words sent")

                # signal end-of-text
                await ws.send_str("__END__")
                print("📤 END sentinel sent — waiting for audio")

                # wait until server closes
                await recv

        except Exception as e:
            print(f"❌ Connection error: {e}")

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("🎉 done")


if __name__ == "__main__":
    sample = (
        "I'm calling regarding your Chase account ending in eight six one three."
    )

    async def main():
        await stream_tts(sample)

    asyncio.run(main())