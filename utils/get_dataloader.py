from dataloaders import EcgDataset2D

def get_dataloader(split="test", batch_size=16):
    # File paths depending on the split
    if split == "test":
        ann_path = "data/val.json"
    else:
        ann_path = "data/train.json"

    mapping_path = "data/class-mapper.json"  # You might use a common class mapping for both

    dataset = EcgDataset2D(ann_path=ann_path, mapping_path=mapping_path)
    dataloader = dataset.get_dataloader(batch_size=batch_size, shuffle=(split != "test"))
    return dataloader
