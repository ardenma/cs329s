from utils.data import LiarDataset
for num_labels in [3, 6]:
    for split in ["train", "validation", "test"]:
        dataset = LiarDataset(split, num_labels=num_labels)
        print(f"Liar dataset ({split}) with n_labels={num_labels} class balance: \n{dataset.get_class_balance(as_tensor=True)}")