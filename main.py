from data_loader import load_datasets
from novel_loader import build_novel_stores
from predictor import predict_example
from threshold_tuning import tune_threshold

def main():
    train_df, test_df = load_datasets()
    stores = build_novel_stores()

    # --- TRAIN PHASE (CALIBRATION) ---
    train_results = []
    train_preds = []

    for _, row in train_df.iterrows():
        result = predict_example(
            row,
            stores[row["book_name"]],
            top_k=8
        )
        train_results.append((result["score"], row["label"]))
        train_preds.append(result)

    threshold, acc = tune_threshold(train_results)
    print("Best threshold:", threshold)
    print("Train accuracy:", acc)

    # --- TEST PHASE ---
    submissions = []

    for _, row in test_df.iterrows():
        result = predict_example(
            row,
            stores[row["book_name"]],
            top_k=8
        )
        label = 1 if result["score"] >= threshold else 0

        submissions.append({
            "id": row["id"],
            "label": label
        })

    # Save submission
    import pandas as pd
    pd.DataFrame(submissions).to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
