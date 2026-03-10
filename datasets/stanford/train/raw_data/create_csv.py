import pandas as pd
from pathlib import Path

### in the folder "raw_data" were folders 'pos' and 'neg' with txt files (I didn't upload them to GitHub), so I needed to create csv ###

def combine_txt_to_csv(folder_paths, output_file, sample_size = None):
    data = []

    for folder in folder_paths:
        path = Path(folder)
        
         # Find all .txt files and slice the list to the sample_size
        all_files = list(path.glob("*.txt"))
        sampled_files = all_files[:sample_size] if sample_size else all_files

        for file_path in sampled_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if folder == "./pos":
                    sentiment = "positive"
                elif folder == "./neg":
                    sentiment = "negative"
                else:
                    sentiment = "unknown"
                # Store the content and sentiment in the data list
                data.append({
                    "review": content,
                    "sentiment": sentiment,
                    "file_name": file_path.name
                })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Successfully created {output_file} with {len(df)} rows.")

# Usage
folders = ["./pos", "./neg"]
combine_txt_to_csv(folders, "combined_data.csv")
combine_txt_to_csv(folders, "combined_data_slice.csv", sample_size=2500)
