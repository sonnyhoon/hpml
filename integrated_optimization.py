import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.utils.prune as prune
import time
import matplotlib.pyplot as plt
import pandas as pd

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

def measure_inference_time(model, dataloader):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(model.device) for key, val in batch.items()}
            with autocast():
                outputs = model(**inputs)
    end_time = time.time()
    return end_time - start_time

# Load model and tokenizer
model_name = "gtfintechlab/FOMC-RoBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to('cuda')  # Move model to GPU

# Apply pruning
parameters_to_prune = (
    (model.roberta.encoder.layer[0].attention.self.query, 'weight'),
    (model.roberta.encoder.layer[0].attention.self.key, 'weight'),
    (model.roberta.encoder.layer[0].attention.self.value, 'weight'),
)
for module, param in parameters_to_prune:
    prune.l1_unstructured(module, param, amount=0.3)
    prune.remove(module, param)

# Apply quantization after pruning
model_quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Prepare dataset and dataloader
texts = ["The economic outlook has improved significantly in recent months."] * 100  # Example texts
dataset = SimpleDataset(tokenizer, texts)
dataloader = DataLoader(dataset, batch_size=10, num_workers=4, shuffle=False)

# Measure performance
inference_time = measure_inference_time(model_quantized, dataloader)

# Output results
print(f"Inference time with optimizations: {inference_time:.4f} seconds")

# Plotting results
fig, ax = plt.subplots()
ax.bar(['Optimized Model'], [inference_time], color='green')
ax.set_ylabel('Inference Time (seconds)')
ax.set_title('Model Performance After Optimizations')
plt.show()

# Save the plot as a PDF
fig.savefig('optimized_performance.pdf')

