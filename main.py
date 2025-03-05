import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import speech_recognition as sr

# Define the paths for the Processing and Complete folders
processing_folder = "Processing"
complete_folder = "Complete"

# Ensure the Complete folder exists
if not os.path.exists(complete_folder):
    os.makedirs(complete_folder)

if not os.path.exists(processing_folder):
    os.makedirs(processing_folder)


class AudioFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Check if the new file is an audio file (e.g., .wav, .mp3)
        if event.is_directory:
            return
        if event.src_path.endswith(".wav") or event.src_path.endswith(".mp3"):
            print(f"New audio file detected: {event.src_path}")
            self.transcribe_audio(event.src_path)


    def transcribe_audio(self, audio_file_path):
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio = recognizer.record(source)
                print("Transcribing audio...")
                text = recognizer.recognize_google(audio)
                print("Transcription complete.")

                # Save the transcribed text to a file
                output_file_name = os.path.basename(audio_file_path).rsplit('.', 1)[0] + ".txt"
                output_file_path = os.path.join(complete_folder, output_file_name)
                with open(output_file_path, "w") as text_file:
                    text_file.write(text)
                print(f"Transcription saved to {output_file_path}")

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")


if __name__ == "__main__":
    event_handler = AudioFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=processing_folder, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()