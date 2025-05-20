from datasets import Dataset

def reading_huggingface_arrow(path="/content/data-00000-of-00243.arrow"):
    ds = Dataset.from_file(path)
    print("Column names:", ds.column_names)
    # print("First row:", ds[0])
    print("Total rows:", len(ds))
