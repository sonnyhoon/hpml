import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import time

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Function to measure inference time with different DataLoader settings
def measure_inference_time(model, dataloader):
    model.eval()
    times = []
    with torch.no_grad():
        start_time = time.time()
        for batch in dataloader:
            inputs = {key: val.to(model.device) for key, val in batch.items()}
            outputs = model(**inputs)
        end_time = time.time()
        times.append(end_time - start_time)
    return sum(times)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gtfintechlab/FOMC-RoBERTa")
model = AutoModelForSequenceClassification.from_pretrained("gtfintechlab/FOMC-RoBERTa")
model.to('cuda')

# Example text and data setup
texts = ["The economic outlook has improved significantly in recent months."] * 1000  # Increased number of examples
dataset = SimpleDataset(tokenizer, texts)

# Different batch sizes and number of workers
batch_sizes = [1, 10, 20, 50, 100]
num_workers = [0, 1, 2, 4, 8]

results = []

for batch_size in batch_sizes:
    for workers in num_workers:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
        avg_time = measure_inference_time(model, dataloader)
        results.append({
            'Batch Size': batch_size,
            'Number of Workers': workers,
            'Inference Time (s)': avg_time
        })

# Convert results to DataFrame and visualize
df = pd.DataFrame(results)
fig, ax = plt.subplots(figsize=(10, 8))
for key, grp in df.groupby('Number of Workers'):
    ax = grp.plot(ax=ax, kind='line', x='Batch Size', y='Inference Time (s)', label=f'Workers {key}', marker='o')

plt.title('Inference Time by Batch Size and Number of Workers')
plt.xlabel('Batch Size')
plt.ylabel('Inference Time (seconds)')
plt.legend(title='Number of Workers')
plt.grid(True)
plt.savefig('data_loading_efficiency.pdf')
plt.show()

