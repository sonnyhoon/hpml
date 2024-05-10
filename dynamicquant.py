import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
model_name = "gtfintechlab/FOMC-RoBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()  # Set to evaluation mode

# Quantize the model on the CPU
model.to('cpu')  # Ensure model is on CPU
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Function to measure inference time
def measure_inference_time(model, inputs, iterations=10):
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.time()
            outputs = model(**inputs)
            end_time = time.time()
            times.append(end_time - start_time)
    return times

# Prepare the input
text = "The economic outlook has improved significantly in recent months."
inputs = tokenizer(text, return_tensors="pt")

# Measure inference time
original_times = measure_inference_time(model, inputs)
quantized_times = measure_inference_time(quantized_model, inputs)

# Calculate average times
original_avg_time = sum(original_times) / len(original_times)
quantized_avg_time = sum(quantized_times) / len(quantized_times)

# Data for plotting
labels = ['Original Model', 'Quantized Model']
times = [original_avg_time, quantized_avg_time]
colors = ['#1f77b4', '#ff7f0e']  # Colors for the bars

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(labels, times, color=colors, width=0.5)
plt.ylabel('Average Inference Time (seconds)')
plt.title('Inference Time Comparison')
for i, v in enumerate(times):
    plt.text(i, v + 0.01, f"{v:.4f}", color='black', ha='center')
plt.tight_layout()
plt.savefig('inference_comparison.pdf')
plt.show()

