import numpy as np
import wave
from io import BytesIO
from openai import OpenAI
from fastrtc import Stream, ReplyOnPause, AdditionalOutputs, wait_for_item
from prompts import TTS_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from rag import RAG


import os
from dotenv import load_dotenv
import gradio as gr
import asyncio

rag = RAG()

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TTS_SAMPLE_RATE = 24000  # OpenAI TTS default
output_queue = asyncio.Queue()


def numpy_to_wav_bytes(sample_rate: int, audio_array: np.ndarray) -> bytes:
    """Convert numpy array to WAV bytes for OpenAI API"""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_array.tobytes())
    return buffer.getvalue()


def pcm_to_numpy(pcm_data: bytes, sample_rate: int) -> tuple[int, np.ndarray]:
    """Convert OpenAI PCM response to FastRTC audio format"""
    audio_array = np.frombuffer(pcm_data, dtype=np.int16)
    return (sample_rate, audio_array)


def voice_assistant_handler(audio: tuple[int, np.ndarray]):
    sample_rate, audio_data = audio
    wav_bytes = numpy_to_wav_bytes(sample_rate, audio_data)

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=("audio.wav", wav_bytes),
        response_format="text",
    )
    print(transcript)
    yield AdditionalOutputs({"role": "user", "content": transcript})
    rag_response = rag.answer(transcript, client, RAG_SYSTEM_PROMPT)
    print(rag_response)
    yield AdditionalOutputs({"role": "assistant", "content": rag_response})

    tts_response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=rag_response,
        response_format="pcm",
        speed=1,
        instructions=TTS_SYSTEM_PROMPT,
    )

    pcm_data = b""
    for chunk in tts_response.iter_bytes():
        pcm_data += chunk
        if len(pcm_data) >= 4800:  # 200ms chunks at 24kHz
            yield pcm_to_numpy(pcm_data[:4800], TTS_SAMPLE_RATE)
            pcm_data = pcm_data[4800:]

    if pcm_data:
        yield pcm_to_numpy(pcm_data, TTS_SAMPLE_RATE)


async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
    return await wait_for_item(output_queue)


def update_chatbot(chatbot: list[dict], response: dict):
    chatbot.append(response)
    return chatbot


chatbot = gr.Chatbot(type="messages")
latest_message = gr.Textbox(type="text", visible=False)
stream = Stream(
    handler=ReplyOnPause(voice_assistant_handler, input_sample_rate=24000),
    modality="audio",
    mode="send-receive",
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    additional_outputs_handler=update_chatbot,
    ui_args={"title": "LLM Voice Chat (Powered by OpenAI and WebRTC ⚡️)"},
)


if __name__ == "__main__":
    # Local development mode
    stream.ui.launch(server_port=7860, debug=True)
