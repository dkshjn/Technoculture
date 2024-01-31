from datasets import DatasetDict, load_dataset

def format_chosen(row):
    formatted_chosen = [
        {"content": row["system"], "role": "system"},
        {"content": row["chosen"], "role": "user"}
    ]
    formatted_row_chosen = ", ".join([f"{{'content': '{item['content']}', 'role': '{item['role']}'}}" for item in formatted_chosen])
    row["formatted_chosen"] = f"[{formatted_row_chosen}]"
    return row

def format_rejected(row):
    formatted_rejected = [
        {"content": row["system"], "role": "system"},
        {"content": row["rejected"], "role": "assistant"}
    ]
    formatted_row_rejected = ", ".join([f"{{'content': '{item['content']}', 'role': '{item['role']}'}}" for item in formatted_rejected])
    row["formatted_rejected"] = f"[{formatted_row_rejected}]"
    return row
                                       
def main():
    """Main function to process and display samples from the dataset."""

    # Load and preprocess dataset
    print("Starting Dataset Loading and Processing...")
    dataset: DatasetDict = (
        load_dataset("argilla/distilabel-intel-orca-dpo-pairs")
    )
    # print(dataset["train"]["chosen"][1])
    # print(dataset["train"]["rejected"][1])
    # print("Formatting chosen.")
    new_dataset = dataset["train"].map(format_chosen)
    new_dataset = new_dataset.map(format_rejected)

    new_dataset.push_to_hub("dkshjn/processed_argilla", token="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")
    # dataset = dataset.map(format_chosen)

    # print(new_dataset["formatted_chosen"][1])
    # print(new_dataset["formatted_rejected"][1])



if __name__ == "__main__":
    main()