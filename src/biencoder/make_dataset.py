import pandas as pd
from datasets import Dataset
from datasets import DatasetDict, Dataset
from datasets import ClassLabel, load_from_disk, load_dataset, concatenate_datasets

class_label_feature = ClassLabel(num_classes=3, names=[0, 1, 2])
class_label_feature
class_label_feature.encode_example(0)


# Amazon ESCI Dataset

esci_class_label_feature = ClassLabel(num_classes=3, names=["Irrelevant", "Substitute", "Exact"])
esci_class_label_feature
esci_class_label_feature.encode_example("Exact")


esci_dataset = load_dataset("tasksource/esci")


ESCI_COL_MAP = {
    "query": "query",
    "product_title": "document",
    "labels": "labels",
}

final_esci_dataset = (
    esci_dataset
    .select_columns(["query", "esci_label", "product_title"])
    .filter(lambda x: x["esci_label"] != "Complement")
    .map(lambda x: dict(x, labels=esci_class_label_feature.encode_example(x["esci_label"])))
    .rename_columns(ESCI_COL_MAP).select_columns(list(ESCI_COL_MAP.values()))
)

for split in ["train", "test"]:
    final_esci_dataset[split] = final_esci_dataset[split].cast_column("labels", class_label_feature)

dataset_save_path = "./data/final_esci_dataset"
# ! rm -rvf {dataset_save_path}
final_esci_dataset.save_to_disk(dataset_save_path)


# Wands Dataset
# get search queries
base_path = "https://github.com/wayfair/WANDS/raw/main/dataset/"
query_df = pd.read_csv(f"{base_path}/query.csv", sep='\t')
product_df = pd.read_csv(f"{base_path}/product.csv", sep='\t', usecols=["product_id", "product_name"])
label_df = pd.read_csv(f"{base_path}/label.csv", sep='\t')

df_dataset = label_df.merge(
    query_df, on="query_id"
).merge(
    product_df, on="product_id"
)


wands_class_label_feature = ClassLabel(num_classes=3, names=["Irrelevant", "Partial", "Exact"])
dataset = (
    Dataset.from_pandas(df_dataset)
    .cast_column("label", wands_class_label_feature)
    # .map(lambda x: dict(x, qir_label=wands_class_label_feature.encode_example(x["label"])))
)
dataset = dataset.train_test_split(test_size=2/5, seed=1337)
dataset_save_path = "./data/wands_dataset"
dataset.save_to_disk(dataset_save_path)
