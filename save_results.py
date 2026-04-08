import os
import re
import pandas as pd

base_dir = "./results_test"
results = {}

# regex to match lines like "ALL pattern: 28.08% (228/812)"
pattern = re.compile(r"ALL ([\w_]+): ([\d.]+)%")

for model_name in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model_name)
    results_file = os.path.join(model_path, "results.txt")

    if not os.path.isfile(results_file):
        continue

    with open(results_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract dataset subcategory accuracies
    subcats = dict(pattern.findall(text))
    if subcats:
        results[model_name] = {k: float(v) for k, v in subcats.items()}

df = pd.DataFrame(results).T 
df = df.sort_index()
# df.rename(columns={'singular_point_detection': 'orientation'}, inplace=True)

# Print table
print("\nModel Subcategory Accuracy Table:\n")
print(df.fillna("-").to_string(float_format=lambda x: f"{x:.2f}%"))

df = df[['pattern','minutiae','orientation','verification','sensor_classification','real_vs_synthetic','acev','tool_retrieval']]
# print(df.columns)

output_path = "model_accuracies.csv"
df.to_csv(output_path, index=True)

print(f"\nSaved table to: {output_path}")
