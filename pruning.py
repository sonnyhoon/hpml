import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.utils.prune as prune
import time

# Function to measure inference time
def measure_inference_time(model, inputs, iterations=10):
    times = []
    model.eval()
    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.time()
            outputs = model(**inputs)
            end_time = time.time()
            times.append(end_time - start_time)
    return times

# Load model and tokenizer
model_name = "gtfintechlab/FOMC-RoBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to('cpu')  # Move model to CPU

# Prepare the input
text = "The economic outlook has improved significantly in recent months."
inputs = tokenizer(text, return_tensors="pt")

# Measure original inference time
original_times = measure_inference_time(model, inputs)
original_avg_time = sum(original_times) / len(original_times)

# Prune the first attention layer's weights by 30%
parameters_to_prune = (
    (model.roberta.encoder.layer[0].attention.self.query, 'weight'),
    (model.roberta.encoder.layer[0].attention.self.key, 'weight'),
    (model.roberta.encoder.layer[0].attention.self.value, 'weight'),
)

for module, param in parameters_to_prune:
    prune.l1_unstructured(module, param, amount=0.3)

# Make pruning permanent
for module, param in parameters_to_prune:
    prune.remove(module, param)

# Measure pruned model inference time
pruned_times = measure_inference_time(model, inputs)
pruned_avg_time = sum(pruned_times) / len(pruned_times)

# Visualization
labels = ['Original', 'Pruned']
times = [original_avg_time, pruned_avg_time]
colors = ['#1f77b4', '#ff7f0e']

plt.figure(figsize=(8, 5))
plt.bar(labels, times, color=colors)
plt.ylabel('Average Inference Time (seconds)')
plt.title('Inference Time: Original vs. Pruned Model')
for i, v in enumerate(times):
    plt.text(i, v + 0.01, f"{v:.4f}", color='black', ha='center')
plt.tight_layout()
plt.savefig('pruning.pdf')
plt.show()

