from datasets import load_dataset, Audio
import os
from tqdm import tqdm
import soundfile as sf

def audio_dataset():
    output_dir = input("Enter output directory (default 'train'): ") or "train"
    n_samples = int(input("Enter number of samples to take (default 100000): ") or 100000)
    dataset_name = input("Enter dataset name (default 'MLCommons/peoples_speech'): ") or "MLCommons/peoples_speech"
    subset_name = input("Enter subset name (default 'clean'): ") or "clean"
    split_name = input("Enter split name (default 'train'): ") or "train"

    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset(dataset_name, subset_name, split=split_name, streaming=True)
    dataset = dataset.cast_column("audio", Audio())
    subset = dataset.take(n_samples)

    for i, sample in enumerate(tqdm(subset, desc="Saving audio")):
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        sf.write(f"{output_dir}/{i:06d}.wav", audio_array, sampling_rate)
