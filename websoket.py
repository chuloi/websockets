import asyncio
import websockets
import wave
import os
import speech_recognition as sr
import socket
import aiohttp
import logging
import json

# Constants
CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16000
FILENAME = "recording.wav"
PREDICTION_URL = "http://localhost:5000/predict" 

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

    async def handle_client(self, websocket, path):
        self.logger.info("Client connected")
        try:
            async for message in websocket:
                if isinstance(message, str):
                    if message == "END":
                        await self.stop_recording()
                        transcription = await self.transcribe_audio()

                        if transcription:
                            self.temp_transcription = transcription
                            if "trợ lý ảo" in transcription.lower():
                                if not self.is_listening:
                                    await self.send_message(websocket, {
                                        "type": "response",
                                        "message": "Xin chào"
                                    })
                                    self.is_listening = True 
                            elif self.is_listening:
                                if self.awaiting_answer:
                                    full_query = f"{self.temp_transcription} {transcription}"
                                    self.logger.info(f"Kết hợp câu lệnh: {full_query}")
                                    prediction = await self.get_prediction(full_query)
                                    self.awaiting_answer = False
                                    self.last_question = ""
                                else:
                                    prediction = await self.get_prediction(transcription)
                                
                                self.logger.info(f"Prediction: {prediction}")
                                if prediction:
                                    if 'message' in prediction:
                                        response_message = prediction['message']
                                        await self.send_message(websocket, {
                                            "type": "response",
                                            "message": response_message
                                        })
                                        self.is_listening = False  
                                    
                                    elif 'question' in prediction:
                                        response_question = prediction['question']
                                        await self.send_message(websocket, {
                                            "type": "question",
                                            "message": response_question
                                        })
                                        self.last_question = response_question
                                        self.awaiting_answer = True
                                        self.temp_transcription = transcription
                                    
                                    else:
                                        response = prediction
                                        response_json = json.dumps({
                                            "type": "prediction",
                                            "data": response
                                        })
                                        self.logger.info(f"Sending response: {response_json}")
                                        await websocket.send(response_json)
                                        self.logger.info(f"Sent prediction to ESP32: {prediction}")
                                        self.is_listening = False 
                                else:
                                    await self.send_error(websocket, "No prediction received")
                            else:
                                self.logger.info("No action required")
                        else:
                            self.logger.error("Transcription failed")
                            await self.send_error(websocket, "Transcription failed")
                elif isinstance(message, bytes):
                    await self.process_audio_data(message) 
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Client connection closed unexpectedly")
        except Exception as e:
            self.logger.error(f"Error in handle_client: {str(e)}")
            error_response = {
                "type": "error",
                "status": "error",
                "message": f"Server error: {str(e)}"
            }
            try:
                await websocket.send(json.dumps(error_response))
            except:
                pass
        finally:
            self.logger.info("Client disconnected")
            await self.stop_recording()

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
            await websocket.send(message_json)
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

async def main():
    server = AudioServer()
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    print(f"Server IP address: {ip_address}")
    print("Server started")

    PORT = int(os.environ.get("PORT", 7777))
    async with websockets.serve(server.handle_client, "0.0.0.0", PORT):  
        await asyncio.Future() 

if __name__ == "__main__":
    asyncio.run(main())
