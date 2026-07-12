import base64
import tempfile

def save_audio_from_base64(audio_base64: str) -> str:
    audio_bytes = base64.b64decode(audio_base64)
    # write to a temporary .wav file so Whisper can read it
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name

import whisper

model = whisper.load_model("small")  # "base" is faster but less accurate for Korean

def transcribe(path: str) -> str:
    result = model.transcribe(path, language="ko")
    return result["text"]

import re
import pandas as pd

def text_to_dataframe(text: str) -> pd.DataFrame:
    numbers = re.findall(r"-?\d+\.?\d*", text)
    values = [float(n) for n in numbers]
    return pd.DataFrame({"value": values})

def compute_stats(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include="number")

    stats = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "mean": numeric_df.mean().to_dict(),
        "std": numeric_df.std().to_dict(),
        "variance": numeric_df.var().to_dict(),
        "min": numeric_df.min().to_dict(),
        "max": numeric_df.max().to_dict(),
        "median": numeric_df.median().to_dict(),
        "mode": {col: df[col].mode().tolist() for col in df.columns},
        "range": (numeric_df.max() - numeric_df.min()).to_dict(),
        "allowed_values": {
            col: sorted(df[col].unique().tolist())
            for col in df.select_dtypes(exclude="number").columns
        },
        "value_range": {
            col: [numeric_df[col].min(), numeric_df[col].max()]
            for col in numeric_df.columns
        },
        "correlation": numeric_df.corr().values.tolist() if numeric_df.shape[1] > 1 else []
    }
    return stats

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

@app.post("/analyze")
def analyze(req: AudioRequest):
    path = save_audio_from_base64(req.audio_base64)
    text = transcribe(path)
    df = text_to_dataframe(text)
    return compute_stats(df)
