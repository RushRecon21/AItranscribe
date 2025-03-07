import io
import logging
import tempfile
import time
import torch
import schedule
from datetime import datetime
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from transformers import AutoTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, GPT2LMHeadModel, \
    GPT2Tokenizer
import whisper
from pydub import AudioSegment
from pydub.effects import normalize, high_pass_filter
from tqdm import tqdm

# Configuration - Update with your paths and IDs
CONFIG = {
    "input_folder_id": "YOUR_INPUT_FOLDER_ID",
    "output_folder_id": "YOUR_OUTPUT_FOLDER_ID",
    "service_account_file": "path/to/service-account.json",
    "whisper_model": "path/to/local/whisper-large",
    "biogpt_model": "path/to/local/custom-biogpt",
    "bart_model": "path/to/local/custom-bart",
    "flan_t5_model": "path/to/local/custom-flan-t5",
    "obsidian_vault": Path("~/Documents/Obsidian Vault").expanduser(),
    "temp_dir": Path(tempfile.gettempdir()) / "med_processor",
    "processed_files": Path(tempfile.gettempdir()) / "med_processor" / "processed.txt",
    "max_tokens": 512,
    "chunk_length_ms": 10 * 60 * 1000,  # 10-minute audio chunks
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("med_processor.log"), logging.StreamHandler()]
)


# Utility Functions
def load_model(model_class, model_path, device="cpu"):
    """Load a model with error handling."""
    try:
        return model_class.from_pretrained(model_path).to(device)
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise


def load_tokenizer(tokenizer_class, model_path):
    """Load a tokenizer with error handling."""
    try:
        return tokenizer_class.from_pretrained(model_path)
    except Exception as e:
        logging.error(f"Failed to load tokenizer from {model_path}: {e}")
        raise


def cleanup_temp_files(*paths):
    """Clean up temporary files safely."""
    for path in paths:
        if path and isinstance(path, Path) and path.exists():
            path.unlink()


# Validate configuration
def validate_config():
    """Validate configuration settings."""
    required = ["input_folder_id", "output_folder_id", "service_account_file",
                "whisper_model", "biogpt_model", "bart_model", "flan_t5_model", "obsidian_vault"]
    for key in required:
        if key not in CONFIG or not CONFIG[key]:
            raise ValueError(
                f"Missing or invalid config key: {key}. Please ensure all required keys are present and correctly configured.")

    for path in ["whisper_model", "biogpt_model", "bart_model", "flan_t5_model", "service_account_file",
                 "obsidian_vault"]:
        if not Path(CONFIG[path]).exists():
            raise ValueError(f"Path does not exist: {CONFIG[path]}. Please ensure the path is correct and accessible.")

    CONFIG["temp_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["processed_files"].parent.mkdir(parents=True, exist_ok=True)


# Google Drive and Docs Manager
class GDriveManager:
    """Manages Google Drive and Docs interactions."""

    def __init__(self):
        self.drive_service = self._authenticate("drive", ["https://www.googleapis.com/auth/drive"])
        self.docs_service = self._authenticate("docs", ["https://www.googleapis.com/auth/documents"])

    def _authenticate(self, service_name, scopes):
        """Authenticate with Google API with retries."""
        for attempt in range(3):
            try:
                creds = service_account.Credentials.from_service_account_file(
                    CONFIG["service_account_file"], scopes=scopes
                )
                return build(service_name, "v3" if service_name == "drive" else "v1", credentials=creds)
            except Exception as e:
                logging.error(f"{service_name} auth attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise Exception(f"Failed to authenticate after 3 attempts: {e}")

    def list_audio_files(self):
        """Fetch audio files from input folder."""
        try:
            results = self.drive_service.files().list(
                q=f"'{CONFIG['input_folder_id']}' in parents and mimeType contains 'audio/'",
                fields="files(id, name)"
            ).execute()
            return results.get("files", [])
        except Exception as e:
            logging.error(f"Failed to list audio files: {e}")
            return []

    def download_file(self, file_id, file_name):
        """Download file from Google Drive with progress."""
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            temp_path = CONFIG["temp_dir"] / file_name
            with open(temp_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                with tqdm(desc=f"Downloading {file_name}", unit="B", unit_scale=True) as pbar:
                    while not done:
                        status, done = downloader.next_chunk()
                        if status:
                            pbar.update(status.resumable_progress)
            logging.info(f"Downloaded {file_name} to {temp_path}")
            return temp_path
        except Exception as e:
            logging.error(f"Download failed: {e}")
            return None

    def create_and_upload_doc(self, title, content):
        """Create a Google Doc and upload it to Drive."""
        try:
            doc = self.docs_service.documents().create(body={"title": title}).execute()
            doc_id = doc["documentId"]
            requests = [{"insertText": {"location": {"index": 1}, "text": content}}]
            self.docs_service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
            self.drive_service.files().update(
                fileId=doc_id, addParents=CONFIG["output_folder_id"], fields="id"
            ).execute()
            logging.info(f"Created and uploaded Google Doc: {title}")
        except Exception as e:
            logging.error(f"Doc creation/upload failed: {e}")

    def sync_to_obsidian(self, content, filename):
        """Write key points and study materials to Obsidian vault in Markdown."""
        obsidian_path = CONFIG["obsidian_vault"] / f"{filename}.md"
        with open(obsidian_path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"Synced {filename} to Obsidian vault")


# Audio Processing with Noise Reduction
class AudioProcessor:
    """Handles audio preprocessing and transcription with Whisper."""

    def __init__(self):
        self.model = load_model(whisper.load_model, CONFIG["whisper_model"],
                                device="cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_audio(self, audio_path):
        """Reduce classroom noise and chatter."""
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = normalize(audio)
            audio = high_pass_filter(audio, cutoff=200)
            preprocessed_path = CONFIG["temp_dir"] / f"preprocessed_{audio_path.name}"
            audio.export(preprocessed_path, format="wav")
            return preprocessed_path
        except Exception as e:
            logging.error(f"Audio preprocessing failed: {e}")
            return audio_path

    def chunk_and_transcribe(self, audio_path):
        """Chunk audio and transcribe with preprocessing."""
        preprocessed_path = self.preprocess_audio(audio_path)
        audio = AudioSegment.from_file(preprocessed_path)
        chunks = [audio[i:i + CONFIG["chunk_length_ms"]] for i in range(0, len(audio), CONFIG["chunk_length_ms"])]
        transcript = []
        for i, chunk in enumerate(tqdm(chunks, desc="Transcribing")):
            with io.BytesIO() as chunk_buffer:
                chunk.export(chunk_buffer, format="wav")
                chunk_buffer.seek(0)
                result = self.model.transcribe(chunk_buffer, fp16=torch.cuda.is_available())
                transcript.append(result["text"])
        cleanup_temp_files(preprocessed_path if preprocessed_path != audio_path else None)
        return " ".join(transcript) if transcript else None


# Text Processing
class TextProcessor:
    """Corrects and summarizes text."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.biogpt_tokenizer = load_tokenizer(GPT2Tokenizer, CONFIG["biogpt_model"])
        self.biogpt_model = load_model(GPT2LMHeadModel, CONFIG["biogpt_model"], device=self.device)
        self.bart_tokenizer = load_tokenizer(AutoTokenizer, CONFIG["bart_model"])
        self.bart_model = load_model(BartForConditionalGeneration, CONFIG["bart_model"], device=self.device)

    def correct_text(self, text):
        """Correct transcription errors with BioGPT."""
        try:
            prompt = f"Correct this medical transcription: {text}"
            inputs = self.biogpt_tokenizer(prompt, return_tensors="pt", max_length=CONFIG["max_tokens"],
                                           truncation=True).to(self.device)
            outputs = self.biogpt_model.generate(
                inputs["input_ids"], max_length=CONFIG["max_tokens"], num_beams=5, early_stopping=True
            )
            corrected = self.biogpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected.replace("Correct this medical transcription: ", "").strip()
        except Exception as e:
            logging.error(f"Text correction failed: {e}")
            return text

    def summarize_text(self, text):
        """Generate a 3-paragraph summary with BART."""
        try:
            summaries = []
            for i in range(0, len(text), CONFIG["max_tokens"]):
                chunk = text[i:i + CONFIG["max_tokens"]]
                inputs = self.bart_tokenizer(chunk, return_tensors="pt", max_length=CONFIG["max_tokens"],
                                             truncation=True).to(self.device)
                summary_ids = self.bart_model.generate(
                    inputs["input_ids"], max_length=150, min_length=50, num_beams=4, early_stopping=True
                )
                summary = self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                if summary.strip():
                    summaries.append(summary)
                if len(summaries) >= 3:
                    break
            return "\n\n".join(summaries) if summaries else "No summary generated."
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return "No summary generated."


# Study Materials Generator
class StudyGenerator:
    """Generates flashcards and practice questions."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = load_tokenizer(AutoTokenizer, CONFIG["flan_t5_model"])
        self.model = load_model(T5ForConditionalGeneration, CONFIG["flan_t5_model"], device=self.device)

    def generate_materials(self, summary):
        """Generate 20 cloze flashcards and 5 practice questions."""
        try:
            # Flashcards
            flash_lines = []
            for i in range(0, len(summary), CONFIG["max_tokens"]):
                chunk = summary[i:i + CONFIG["max_tokens"]]
                prompt_flash = (
                    f"Generate 20 challenging medical school Anki cloze-deletion flashcards from this summary "
                    f"in the format '{{{{c1::term}}}} - definition'. Summary: {chunk}"
                )
                inputs_flash = self.tokenizer(prompt_flash, return_tensors="pt", max_length=CONFIG["max_tokens"],
                                              truncation=True).to(self.device)
                flash_ids = self.model.generate(inputs_flash["input_ids"], max_length=2048, num_beams=5,
                                                early_stopping=True)
                flashcards = self.tokenizer.decode(flash_ids[0], skip_special_tokens=True)
                flash_lines.extend(
                    [line.strip() for line in flashcards.split("\n") if "{{c1::" in line][:20 - len(flash_lines)]])
                if len(flash_lines) >= 20:
                    break

            # Questions
            question_lines = []
            for i in range(0, len(summary), CONFIG["max_tokens"]):
                chunk = summary[i:i + CONFIG["max_tokens"]]
                prompt_questions = (
                    f"Generate 5 Amboss/UWorld-style practice questions with detailed explanations from this summary "
                    f"in the format 'Q: [question]\nA: [answer]\nExplanation: [explanation]'. Summary: {chunk}"
                )
                inputs_questions = self.tokenizer(prompt_questions, return_tensors="pt",
                                                  max_length=CONFIG["max_tokens"], truncation=True).to(self.device)
                question_ids = self.model.generate(inputs_questions["input_ids"], max_length=2048, num_beams=5,
                                                   early_stopping=True)
                questions = self.tokenizer.decode(question_ids[0], skip_special_tokens=True)
                question_lines.extend(
                    [f"Q:{q.strip()}" for q in questions.split("\nQ:")[1:6 - len(question_lines)] if q.strip()])
                if len(question_lines) >= 5:
                    break

            # Google Doc (Anki-ready)
            gdoc_flashcards = "\n".join(flash_lines)
            gdoc_questions = "\n".join(question_lines)

            # Obsidian (Markdown with anki code blocks)
            obsidian_flashcards = "```anki\n" + "\n".join(flash_lines) + "\n```"
            obsidian_questions = "\n".join(
                [f"### Question {i + 1}\n```anki\n{q}\n```" for i, q in enumerate(question_lines)])

            return {
                "gdoc": {"flashcards": gdoc_flashcards, "questions": gdoc_questions},
                "obsidian": {"flashcards": obsidian_flashcards, "questions": obsidian_questions}
            }
        except Exception as e:
            logging.error(f"Study material generation failed: {e}")
            return {"gdoc": {"flashcards": "", "questions": ""}, "obsidian": {"flashcards": "", "questions": ""}}


# Processing Workflow
def process_audio_files():
    """Process audio files from Google Drive."""
    drive = GDriveManager()
    audio_proc = AudioProcessor()
    text_proc = TextProcessor()
    study_gen = StudyGenerator()

    processed_ids = set()
    if CONFIG["processed_files"].exists():
        with open(CONFIG["processed_files"], "r", encoding="utf-8") as f:
            processed_ids = set(f.read().splitlines())
    audio_files = drive.list_audio_files()

    for audio_file in audio_files:
        file_id = audio_file["id"]
        if file_id in processed_ids:
            logging.info(f"Skipping {audio_file['name']}")
            continue

        logging.info(f"Processing {audio_file['name']}")
        audio_path = drive.download_file(file_id, audio_file["name"])
        if not audio_path:
            continue

        try:
            # Transcribe with noise reduction
            transcript = audio_proc.chunk_and_transcribe(audio_path)
            if not transcript:
                logging.error("Transcription empty")
                continue

            # Correct and summarize
            corrected = text_proc.correct_text(transcript)
            summary = text_proc.summarize_text(corrected)
            materials = study_gen.generate_materials(summary)

            # Extract key points
            key_points = "\n".join([f"- {p.strip()}" for p in summary.split(". ")[:3] if p.strip()])

            # File naming: Today's date.time (e.g., 2025-03-06.1430)
            now = datetime.now()
            filename = now.strftime("%Y-%m-%d.%H%M")

            # Google Doc output
            gdoc_content = (
                f"# Summary\n{summary}\n\n"
                f"# Key Points\n{key_points}\n\n"
                f"# Study Materials\n## Flashcards\n{materials['gdoc']['flashcards']}\n\n## Practice Questions\n{materials['gdoc']['questions']}\n\n"
                f"# Transcript Refined\n{corrected}"
            )

            # Obsidian output
            obsidian_content = (
                f"# Key Points\n{key_points}\n\n"
                f"# Study Materials\n## Flashcards\n{materials['obsidian']['flashcards']}\n\n## Practice Questions\n{materials['obsidian']['questions']}"
            )

            # Upload and sync
            drive.create_and_upload_doc(filename, gdoc_content)
            drive.sync_to_obsidian(obsidian_content, filename)

            # Track processed file
            processed_ids.add(file_id)
            with open(CONFIG["processed_files"], "a", encoding="utf-8") as f:
                f.write(f"{file_id}\n")
            logging.info(f"Completed {audio_file['name']}")

        finally:
            # Robust cleanup using utility function
            temp_files = CONFIG["temp_dir"].glob("chunk_*.wav") + CONFIG["temp_dir"].glob("preprocessed_*")
            cleanup_temp_files(audio_path, *temp_files)


# Scheduling Logic
def run_if_in_time_window():
    """Run processing if current time is between 8 AM and 7 PM."""
    now = datetime.now()
    current_hour = now.hour
    if 8 <= current_hour < 19:  # 8 AM to 7 PM
        logging.info("Within operating hours, scanning Google Drive...")
        process_audio_files()
    else:
        logging.info("Outside operating hours (8 AM - 7 PM), skipping scan.")


# Main Function with Scheduling
def main():
    """Setup and run the scheduled processing."""
    try:
        validate_config()
        logging.info("Starting scheduled audio processing...")
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        raise

    # Schedule to run every 15 minutes
    schedule.every(15).minutes.do(run_if_in_time_window)

    # Run indefinitely
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            logging.error(f"Scheduling error: {e}")
            time.sleep(60)  # Prevent tight loop on failure


if __name__ == "__main__":
    main()
