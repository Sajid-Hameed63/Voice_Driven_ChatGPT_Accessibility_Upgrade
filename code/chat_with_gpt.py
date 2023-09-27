import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
import whisper
import torch
import os
import openai
import soundfile as sf
import threading
import numpy as np
import gtts
from pydub import AudioSegment
from pydub.playback import play
import pygame
from playsound import playsound


class AudioRecorder:
    def __init__(self):
        self.sr = 16000  # Sample rate
        self.audio_data = []
        self.recording = False
        self.record_thread = None

    def record_audio(self):
        self.audio_data = []  # Clear any previous data
        self.recording = True
        print("Recording started. Press Enter to stop...")
        while self.recording:
            myrecording = sd.rec(1024, samplerate=self.sr, channels=2)
            sd.wait()
            self.audio_data.extend(myrecording)

    def stop_recording(self):
        self.recording = False
        print("Recording stopped")
    
    def save_audio(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        recordings_folder = "recordings"
        os.makedirs(recordings_folder, exist_ok=True)
        file_path = os.path.join(recordings_folder, f'recording_{timestamp}.wav')
        write(file_path, self.sr, np.array(self.audio_data))  # Convert to NumPy array before saving
        # print(f"Recording saved at: {file_path}")
        return file_path
      	

    def record_until_stop(self):
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.start()
        input("Press Enter to stop recording...")
        self.stop_recording()
        self.record_thread.join()
        return self.save_audio()

def recording_from_microphone():
    recorder = AudioRecorder()
    return recorder.record_until_stop()
 

def load_whisper_model(device):
    model = whisper.load_model("small.en") # this statement will load the model using internet [small.en, base.en, tiny.en, medium.en, large]
    # model = "path/to/locally_speech_recognition_model.pt" # here you can specify the path for already downloaded speech recognition model
    return model.to(device)

def check_cuda_availability():
    """Check if CUDA (GPU) is available; otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def speech_to_text(audio_data):
    device = check_cuda_availability()
    model = load_whisper_model(device)
    try:
        result = model.transcribe(audio_data, language='en') # transcribing the audio data using the Whisper ASR by Open AI
        recognized_text = result['text'] # retrieving the text
        return recognized_text
        
    except Exception as e:
        print(f"An error occurred while processing recording': {e}")


def recognized_text_to_chat_gpt(prompt, api_key):
    openai.api_key = api_key
    
    # Call the OpenAI API to get a response
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choose an engine
        prompt=prompt,
        max_tokens=2048,  # Adjust this based on your desired response length
    )
    
    # Extract and return the response
    chat_gpt_response = response.choices[0].text
    return chat_gpt_response

    
def text_to_speech(text, audio_name):
    # make request to google to get synthesis
    tts = gtts.gTTS(text)

    recordings_folder = "recordings"
    os.makedirs(recordings_folder, exist_ok=True)

    file_path = os.path.join(recordings_folder, f"{audio_name}_chat_gpt_response.wav")

    # save the audio file
    tts.save(file_path)

    return file_path


    

    
if __name__ == "__main__":

    while True:
        query_recording_path = recording_from_microphone()
        recognized_text = speech_to_text(audio_data=query_recording_path)
        # print(f"User Query ==> {recognized_text}")
        open_ai_key = "" # set your open-ai key, paid account needed.
        chat_gpt_response = recognized_text_to_chat_gpt(prompt=recognized_text, api_key=open_ai_key)
        # print(f"ChatGPT's Response ==> {chat_gpt_response}")
        basename = os.path.basename(query_recording_path)
        chat_gpt_response_audio_name_without_ext = os.path.splitext(basename)[0]
        path_chat_gpt_reponse_audio_path = text_to_speech(text=chat_gpt_response, audio_name=chat_gpt_response_audio_name_without_ext)
        playsound(path_chat_gpt_reponse_audio_path)
