import torch
from torch.cuda.amp import autocast
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd
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

# Function to measure inference time with mixed precision
def measure_inference_time_mixed_precision(model, inputs, iterations=10):
    times = []
    model.eval()
    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.time()
            with autocast():
                outputs = model(**inputs)
            end_time = time.time()
            times.append(end_time - start_time)
    return times

# Load model and tokenizer
model_name = "gtfintechlab/FOMC-RoBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to('cuda')  # Assume using GPU

# Prepare the input
text = "The economic outlook has improved significantly in recent months."
inputs = tokenizer(text, return_tensors="pt").to('cuda')

# Measure original and mixed precision inference times
original_times = measure_inference_time(model, inputs)
mixed_precision_times = measure_inference_time_mixed_precision(model, inputs)

# Calculate average times
original_avg_time = sum(original_times) / len(original_times)
mixed_precision_avg_time = sum(mixed_precision_times) / len(mixed_precision_times)

# Data for plotting
labels = ['Original', 'Mixed Precision']
times = [original_avg_time, mixed_precision_avg_time]
colors = ['#007acc', '#d62728']

# Create a DataFrame for saving to CSV
df = pd.DataFrame(list(zip(labels, times)), columns=['Model', 'Average Inference Time'])
df.to_csv('inference_times.csv', index=False)

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(labels, times, color=colors, width=0.4)
plt.ylabel('Average Inference Time (seconds)')
plt.title('Inference Time Comparison')
plt.ylim(0, max(times) * 1.2)  # Scale y-axis for better visual representation

# Add text annotations on bars
for i, v in enumerate(times):
    plt.text(i, v + 0.01, f"{v:.4f} s", color='black', ha='center')

plt.tight_layout()
plt.savefig('mix_inference_comparison.pdf')  # Save the figure as a PDF
plt.show()

