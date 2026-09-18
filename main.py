import sys
import time
import numpy as np
import ollama

from audio_io import AudioIO
from vad import SileroVAD
from stt import Transcriber
from tts import Synthesizer
from tools import execute_clinic_tool

SYSTEM_PROMPT = """You are a warm, direct AI receptionist for BrightSmile Dental Clinic.
IMPORTANT RULES:
1. Keep replies extremely brief (1-2 sentences maximum).
2. Answer based on business hours (Mon-Fri 9 AM - 5 PM) and available services.
3. Be conversational and natural. Do NOT use bullet points or formatting.
"""

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_clinic_info",
            "description": "Check clinic hours, available services, or open appointment slots.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a dental appointment slot for the patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_slot": {"type": "string", "description": "The requested time slot, e.g. '2:00 PM'"},
                    "name": {"type": "string", "description": "Name of the patient"}
                },
                "required": ["time_slot"]
            },
        },
    }
]

def main():
    print("==================================================")
    print("🚀 Initializing Voice Receptionist on Mac M1...")
    print("==================================================")

    audio = AudioIO()
    # 0.35 is much more sensitive and reliable for standard headsets / mics
    vad = SileroVAD(threshold=0.35)
    stt = Transcriber(model_size="base.en")
    tts = Synthesizer()

    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    audio.start()
    print("\n✅ System READY! Wearing headphones is recommended to prevent mic echo.")
    print("🗣️  Say: 'Hi, what time are you open today?' or 'Can I book a checkup?'\n")

    # Initial Agent Greeting
    greeting = "Hello! Thanks for calling BrightSmile Clinic. How can I help you today?"
    print(f"🤖 AI: {greeting}")
    audio.play_audio(tts.speak(greeting))
    
    # Allow greeting to finish playing before opening user speech window
    time.sleep(3.5)

    speech_buffer = []
    silence_frames = 0
    is_recording_speech = False
    
    # 512 samples at 16kHz = 32ms per chunk
    # 16 frames * 32ms ~= 512ms of silence triggers end-of-speech
    SILENCE_THRESHOLD_FRAMES = 16

    print("\n🎧 Listening for your voice now...")

    try:
        while True:
            chunk = audio.read_mic_chunk()
            has_voice = vad.is_speech(chunk)

            # 1. Barge-in / Interruption handling
            if has_voice and audio.is_playing:
                print("\n⚡ [Barge-In] Caller interrupted! Halting playback...")
                audio.stop_playback()
                speech_buffer.clear()
                vad.reset_states()

            # 2. Voice Detected -> Start buffering & give visual feedback
            if has_voice:
                if not is_recording_speech:
                    print("🎙️ [Hearing voice...]", end="", flush=True)
                else:
                    print(".", end="", flush=True)
                is_recording_speech = True
                silence_frames = 0
                speech_buffer.append(chunk)

            # 3. Speech ongoing, but currently a silence frame
            elif is_recording_speech:
                silence_frames += 1
                speech_buffer.append(chunk)

                # End of speech reached
                if silence_frames > SILENCE_THRESHOLD_FRAMES:
                    print(" [Speech ended]")
                    is_recording_speech = False
                    silence_frames = 0
                    
                    full_audio = np.concatenate(speech_buffer)
                    speech_buffer.clear()
                    vad.reset_states()

                    # Only process if buffer is longer than ~0.35 seconds
                    if len(full_audio) > 16000 * 0.35:
                        print("⏳ Transcribing speech...")
                        t0 = time.time()
                        user_text = stt.transcribe(full_audio)
                        print(f"👤 Caller ({round((time.time() - t0)*1000)}ms): {user_text}")

                        if not user_text.strip():
                            print("🎧 Listening...")
                            continue

                        # Add to conversation context
                        conversation_history.append({"role": "user", "content": user_text})

                        # 4. LLM Query via Ollama
                        t_llm0 = time.time()
                        response = ollama.chat(
                            model="qwen2.5:3b",
                            messages=conversation_history,
                            tools=TOOLS_SPEC
                        )
                        message = response.get("message", {})

                        # 5. Check for Function Calling
                        if message.get("tool_calls"):
                            for tool_call in message["tool_calls"]:
                                print(f"🔧 Tool Triggered: {tool_call['function']['name']}")
                                tool_result = execute_clinic_tool(tool_call)
                                conversation_history.append(message)
                                conversation_history.append({
                                    "role": "tool",
                                    "content": tool_result
                                })
                            
                            # Re-prompt LLM with tool output
                            second_resp = ollama.chat(
                                model="qwen2.5:3b",
                                messages=conversation_history
                            )
                            assistant_reply = second_resp["message"]["content"]
                        else:
                            assistant_reply = message.get("content", "")

                        conversation_history.append({"role": "assistant", "content": assistant_reply})
                        print(f"🤖 AI ({round((time.time() - t_llm0)*1000)}ms): {assistant_reply}")

                        # 6. Synthesize & Play back
                        audio_out = tts.speak(assistant_reply)
                        audio.play_audio(audio_out)
                        print("\n🎧 Listening...")

    except KeyboardInterrupt:
        print("\n🛑 Stopping Voice Receptionist...")
    finally:
        audio.close()
        print("Done.")

if __name__ == "__main__":
    main()