import json
import os
import os.path as osp
from glob import glob

import pandas as pd

# Define ECG classes
classes = ["N", "V", "\\", "R", "L", "A", "!", "E"]
lead = "MLII"
extension = "png"  # or `npy` for 1D
val_size = 0.1  # [0, 1]
random_state = 7

# Updated data path based on your actual file structure
data_path = osp.abspath(r"K:/Git/ecg-classification/data/2D//*/*/*/*.{}".format(extension))
output_path = "/".join(data_path.split("/")[:-5])

if __name__ == "__main__":
    dataset = []
    files = glob(data_path)

    print("Looking for files at:", data_path)
    print("Found files:", len(files))

    for file in files:
        parts = file.replace("\\", "/").split("/")
        if len(parts) < 5:
            print(f"Skipping malformed path: {file}")
            continue

        name = parts[-4]  # "100"
        lead_ = parts[-3]  # "MLII"
        label = parts[-2]  # "A"
        filename = parts[-1]  # "2044.png"

        dataset.append(
            {
                "name": name,
                "lead": lead_,
                "label": label,
                "filename": osp.splitext(filename)[0],
                "path": file,
            },
        )

    data = pd.DataFrame(dataset)

    if data.empty:
        raise ValueError("No data was loaded. Please check the path and file structure.")

    if "lead" not in data.columns:
        print("Available columns:", data.columns)
        raise KeyError("'lead' column not found in DataFrame")

    data = data[data["lead"] == lead]
    data = data[data["label"].isin(classes)]
    data = data.sample(frac=1, random_state=random_state)

    val_ids = []
    for cl in classes:
        val_ids.extend(
            data[data["label"] == cl]
            .sample(frac=val_size, random_state=random_state)
            .index,
        )

    val = data.loc[val_ids, :]
    train = data[~data.index.isin(val.index)]

    train.to_json(osp.join(output_path, "train.json"), orient="records")
    val.to_json(osp.join(output_path, "val.json"), orient="records")

    class_map = {label: idx for idx, label in enumerate(sorted(train.label.unique()))}

    with open(osp.join(output_path, "class-mapper.json"), "w") as file:
        file.write(json.dumps(class_map, indent=1))

    print("✅ Annotation generation complete.")
