import pandas as pd
from datasets import load_dataset

hotpot_qa = load_dataset("hotpot_qa", "fullwiki")
print(hotpot_qa)

# train = pd.DataFrame(hotpot_qa['train'])

def save_hq_dataset(data_split):
    if data_split == "dev":
        data_split_name = "validation"
    else:
        data_split_name = data_split
    data = pd.DataFrame(hotpot_qa[data_split_name])
    print(data["question"].str.len().nlargest())
    data.to_csv(f"hotpot_{data_split}.csv", index=False)
    data["answer"] = data["answer"].apply(lambda x: [x])
    data[["answer", "question"]].to_json(f"HQ-open.train-{data_split}.jsonl", orient="records", lines=True)

save_hq_dataset("train")
save_hq_dataset("dev")
save_hq_dataset("test")
