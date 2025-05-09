from datasets import load_dataset

def download_hf():
  dataset_name = input("Enter the dataset name: ")
  subset_name = input("Enter subset name: ")
  ds = load_dataset(dataset_name, name=subset_name)
  for split in ds:
    ds[split].to_pandas().to_csv(f"{subset_name}.csv", index=False)
