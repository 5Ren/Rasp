import pandas as pd
import numpy as np
import os
from glob import glob

def generate_histogram_by_square(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    bins = np.arange(0, 310, 10)

    hist_data = df.groupby('Square_number')['Diameter_nm'].apply(lambda x: np.histogram(x, bins=bins)[0])
    hist_df = pd.DataFrame(hist_data.tolist(), index=hist_data.index, columns=[f'{i}-{i+10}nm' for i in bins[:-1]])

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_Diameter_Histogram.csv")
    hist_df.to_csv(output_path)
    print(f"✔ Saved: {output_path}")
    return output_path

def process_all_csv_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_files = glob(os.path.join(input_dir, "*.csv"))
    for file in csv_files:
        try:
            generate_histogram_by_square(file, output_dir)
        except Exception as e:
            print(f"✖ Error processing {file}: {e}")

# 使用例：
process_all_csv_in_directory("./data_1003", "./output_data_1003")
