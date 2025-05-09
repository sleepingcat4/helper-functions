from datasets import Dataset, DatasetDict
from huggingface_hub import login
from pathlib import Path

def upload_hub():
    repo_id = input("Enter the Hugging Face repo ID (e.g., username/repo-name): ")
    csv_filename = input("Enter the CSV file to upload: ")
    dict_name = input("Enter the DatasetDict key name: ")
    
    ds = Dataset.from_csv(csv_filename)
    dataset_dict = DatasetDict({dict_name: ds})
    dataset_dict.push_to_hub(repo_id)
