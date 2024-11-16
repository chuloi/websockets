# server.py
import os
import asyncio
import wave
import json
import logging
import socket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import speech_recognition as sr
import aiohttp

# Constants
CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16000
FILENAME = "recording.wav"
PREDICTION_URL = os.getenv("PREDICTION_URL", "http://your-prediction-service-url/predict") 

app = FastAPI()

class AudioServer:
    def __init__(self):
        self.wav_file = None
        self.recording = False
        self.buffer = bytearray()
        self.recognizer = sr.Recognizer()
        self.is_listening = False  
        self.last_question = ""
        self.awaiting_answer = False

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def process_audio_data(self, data):
        if not self.recording:
            await self.start_recording()
        self.buffer.extend(data)
        if len(self.buffer) >= 4096:
            self.wav_file.writeframes(self.buffer)
            self.buffer.clear()

    async def start_recording(self):
        if not self.recording:
            self.logger.info("Starting recording")
            if os.path.exists(FILENAME):
                os.remove(FILENAME)
            self.wav_file = wave.open(FILENAME, 'wb')
            self.wav_file.setnchannels(CHANNELS)
            self.wav_file.setsampwidth(SAMPLE_WIDTH)
            self.wav_file.setframerate(SAMPLE_RATE)
            self.recording = True

    async def stop_recording(self):
        if self.recording:
            self.logger.info("Stopping recording")
            if self.buffer:
                self.wav_file.writeframes(self.buffer)
                self.buffer.clear()
            self.wav_file.close()
            self.wav_file = None
            self.recording = False

    async def transcribe_audio(self):
        self.logger.info("Transcribing audio...")
        try:
            if not os.path.exists(FILENAME) or os.path.getsize(FILENAME) == 0:
                self.logger.error(f"Error: {FILENAME} does not exist or is empty")
                return None

            with sr.AudioFile(FILENAME) as source:
                audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio, language="vi-VN")

            self.logger.info(f"Transcription: {text}")
            return text
        except sr.UnknownValueError:
            self.logger.error("Could not understand audio")
        except sr.RequestError as e:
            self.logger.error(f"Google Speech Recognition error; {e}")
        except Exception as e:
            self.logger.error(f"Error during transcription: {e}")
        return None

    async def get_prediction(self, text):
        payload = {"text": text}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(PREDICTION_URL, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        self.logger.error(f"Error getting prediction: {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error in get_prediction: {str(e)}")
            return None

    async def send_message(self, websocket, message):
        try:
            message_json = json.dumps(message)
            await websocket.send_text(message_json)
            self.logger.info(f"Sent message to client: {message_json}")
        except Exception as e:
            self.logger.error(f"Error sending message to client: {e}")

    async def send_error(self, websocket, error_message):
        error_response = {
            "type": "error",
            "status": "error",
            "message": error_message
        }
        await self.send_message(websocket, error_response)

audio_server = AudioServer()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_server.logger.info("Client connected")
    try:
        while True:
            message = await websocket.receive()
            if isinstance(message, str):
                if message == "END":
                    await audio_server.stop_recording()
                    transcription = await audio_server.transcribe_audio()

                    if transcription:
                        audio_server.temp_transcription = transcription
                        if "trợ lý ảo" in transcription.lower():
                            if not audio_server.is_listening:
                                await audio_server.send_message(websocket, {
                                    "type": "response",
                                    "message": "Xin chào"
                                })
                                audio_server.is_listening = True 
                        elif audio_server.is_listening:
                            if audio_server.awaiting_answer:
                                full_query = f"{audio_server.temp_transcription} {transcription}"
                                audio_server.logger.info(f"Kết hợp câu lệnh: {full_query}")
                                prediction = await audio_server.get_prediction(full_query)
                                audio_server.awaiting_answer = False
                                audio_server.last_question = ""
                            else:
                                prediction = await audio_server.get_prediction(transcription)
                            
                            audio_server.logger.info(f"Prediction: {prediction}")
                            if prediction:
                                if 'message' in prediction:
                                    response_message = prediction['message']
                                    await audio_server.send_message(websocket, {
                                        "type": "response",
                                        "message": response_message
                                    })
                                    audio_server.is_listening = False  
                                
                                elif 'question' in prediction:
                                    response_question = prediction['question']
                                    await audio_server.send_message(websocket, {
                                        "type": "question",
                                        "message": response_question
                                    })
                                    audio_server.last_question = response_question
                                    audio_server.awaiting_answer = True
                                    audio_server.temp_transcription = transcription
                                
                                else:
                                    response = prediction
                                    response_json = json.dumps({
                                        "type": "prediction",
                                        "data": response
                                    })
                                    audio_server.logger.info(f"Sending response: {response_json}")
                                    await websocket.send_text(response_json)
                                    audio_server.logger.info(f"Sent prediction to ESP32: {prediction}")
                                    audio_server.is_listening = False 
                            else:
                                await audio_server.send_error(websocket, "No prediction received")
                        else:
                            audio_server.logger.info("No action required")
                    else:
                        audio_server.logger.error("Transcription failed")
                        await audio_server.send_error(websocket, "Transcription failed")
            elif isinstance(message, bytes):
                await audio_server.process_audio_data(message)
    except WebSocketDisconnect:
        audio_server.logger.info("Client disconnected")
        await audio_server.stop_recording()
    except Exception as e:
        audio_server.logger.error(f"Error in websocket_endpoint: {str(e)}")
        error_response = {
            "type": "error",
            "status": "error",
            "message": f"Server error: {str(e)}"
        }
        try:
            await websocket.send_text(json.dumps(error_response))
        except:
            pass

@app.get("/")
async def read_root():
    return JSONResponse(content={"message": "WebSocket server is running"})

if __name__ == "__main__":
    import uvicorn
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print(f"Server IP address: {ip_address}")
    print("Server started")
    uvicorn.run("websoket:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7777)), log_level="info")
