from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
)
import torch
import torchaudio
import tempfile
import os
import ffmpeg
from typing import Optional

app = FastAPI()
model = None
tokenizer = None
feature_extractor = None
config = None

WHISPER_FEAT_CFG = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}

@app.on_event("startup")
async def load_model():
    global model, tokenizer, feature_extractor, config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = AutoConfig.from_pretrained(
        "zai-org/GLM-ASR-Nano-2512",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "zai-org/GLM-ASR-Nano-2512",
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "zai-org/GLM-ASR-Nano-2512",
        trust_remote_code=True
    )
    feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)
    model.eval()
    print(f"Model loaded on device: {device}")

@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "glm-nano-2512", "object": "model"}]}

def get_audio_token_length(seconds, merge_factor=2):
    def get_T_after_cnn(L_in, dilation=1):
        for padding, kernel_size, stride in eval("[(1,3,1)] + [(1,3,2)]"):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out
    
    mel_len = int(seconds * 100)
    audio_len_after_cnn = get_T_after_cnn(mel_len)
    audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1
    audio_token_num = min(audio_token_num, 1500 // merge_factor)
    return audio_token_num

def build_prompt(audio_path: str, merge_factor: int, chunk_seconds: int = 30):
    wav, sr = torchaudio.load(audio_path)
    wav = wav[:1, :]
    if sr != feature_extractor.sampling_rate:
        wav = torchaudio.transforms.Resample(sr, feature_extractor.sampling_rate)(wav)
    
    tokens = []
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\n")
    
    audios = []
    audio_offsets = []
    audio_length = []
    chunk_size = chunk_seconds * feature_extractor.sampling_rate
    
    for start in range(0, wav.shape[1], chunk_size):
        chunk = wav[:, start : start + chunk_size]
        mel = feature_extractor(
            chunk.numpy(),
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )["input_features"]
        audios.append(mel)
        seconds = chunk.shape[1] / feature_extractor.sampling_rate
        num_tokens = get_audio_token_length(seconds, merge_factor)
        tokens += tokenizer.encode("<|begin_of_audio|>")
        audio_offsets.append(len(tokens))
        tokens += [0] * num_tokens
        tokens += tokenizer.encode("<|end_of_audio|>")
        audio_length.append(num_tokens)
    
    if not audios:
        raise ValueError("Audio is empty or failed to load")
    
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\nPlease transcribe this audio into text")
    tokens += tokenizer.encode("<|assistant|>")
    tokens += tokenizer.encode("\n")
    
    batch = {
        "input_ids": torch.tensor([tokens], dtype=torch.long),
        "audios": torch.cat(audios, dim=0),
        "audio_offsets": [audio_offsets],
        "audio_length": [audio_length],
        "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
    }
    return batch

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = "auto"
):
    if not model or not tokenizer or not feature_extractor or not config:
        raise HTTPException(500, "Model not loaded")

    # Save and convert to WAV (16kHz for GLM)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        input_stream = ffmpeg.input('pipe:0')
        out_stream = ffmpeg.output(
            input_stream, tmp.name, format='wav', acodec='pcm_s16le', ar=16000
        )
        ffmpeg.run(out_stream, input=content, overwrite_output=True, quiet=True)
        audio_path = tmp.name

    try:
        device = next(model.parameters()).device
        batch = build_prompt(audio_path, config.merge_factor)
        
        # Prepare inputs
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audios = batch["audios"].to(device)
        
        model_inputs = {
            "inputs": input_ids,
            "attention_mask": attention_mask,
            "audios": audios.to(torch.bfloat16),
            "audio_offsets": batch["audio_offsets"],
            "audio_length": batch["audio_length"],
        }
        prompt_len = input_ids.size(1)
        
        # Inference
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        
        # Decode
        transcript_ids = generated[0, prompt_len:].cpu().tolist()
        transcript = tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
        
        return JSONResponse({
            "text": transcript or "[Empty transcription]"
        })
    finally:
        os.unlink(audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
