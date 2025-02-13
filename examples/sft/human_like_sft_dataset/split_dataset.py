from datasets import load_dataset

dataset = load_dataset("phunguyen01/human-like-sft-dataset")
dataset = dataset['train'].train_test_split(test_size=0.1)
dataset.push_to_hub("tuenguyen/human-like-sft-dataset-split")
print(dataset)