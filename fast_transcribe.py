import nemo.collections.asr as nemo_asr
import os
import pandas as pd
from tqdm import tqdm

def transcribe_wavs(wav_dir, output_csv="transcribe.csv"):
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])
    
    results = []
    for wav_file in tqdm(wav_files, desc="Transcribing"):
        full_path = os.path.join(wav_dir, wav_file)
        transcription = asr_model.transcribe([full_path])[0].text
        results.append({"filename": wav_file, "text": transcription})
    
    pd.DataFrame(results).to_csv(output_csv, index=False)
