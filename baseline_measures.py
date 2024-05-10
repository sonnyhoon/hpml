import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd

# Load the model and tokenizer
model_name = "gtfintechlab/FOMC-RoBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Define the text and tokenize
text = "The economic outlook has improved significantly in recent months."
inputs = tokenizer(text, return_tensors="pt")

# Move model to GPU if available
# if torch.cuda.is_available():
model.cuda()
inputs = inputs.to('cuda')

# Function to perform inference and measure time
def measure_inference_time(model, inputs, iterations=100):
    times = []
    for _ in range(iterations):
        start_time = time.time()
        with torch.no_grad():
            logits = model(**inputs).logits
        end_time = time.time()
        times.append(end_time - start_time)
    return times

# Measure inference time
inference_times = measure_inference_time(model, inputs)

# Save the times to a CSV (optional)
import pandas as pd
df = pd.DataFrame(inference_times, columns=['Inference Time'])
df.to_csv('inference_times.csv', index=False)

# Load data
df = pd.read_csv('inference_times.csv')

# Set a style
plt.style.use('seaborn-darkgrid')

# Define color and style
color = "#2B4E72"  # A deep blue shade
marker_style = dict(linestyle='-', linewidth=1.5, marker='o', markersize=5, color=color)

# Create figure and axes
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax[0].hist(df['Inference Time'], bins=30, color=color, alpha=0.75)
ax[0].set_title('Distribution of Inference Times', fontsize=14, fontweight='bold')
ax[0].set_xlabel('Inference Time (seconds)', fontsize=12)
ax[0].set_ylabel('Frequency', fontsize=12)

# Line plot
ax[1].plot(df['Inference Time'], **marker_style)
ax[1].set_title('Inference Time Over Iterations', fontsize=14, fontweight='bold')
ax[1].set_xlabel('Iteration', fontsize=12)
ax[1].set_ylabel('Inference Time (seconds)', fontsize=12)

# Fine-tuning
for axis in ax:
    axis.tick_params(labelsize=10)
    axis.set_facecolor('#F5F5F5')  # Light gray background for plot area
    for spine in axis.spines.values():
        spine.set_visible(False)

fig.tight_layout(pad=3.0)
plt.show()


fig.savefig('inference_visualizations.pdf', format='pdf', dpi=300)

