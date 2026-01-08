def collect_errors(df, predictions):
    errors = []

    for (_, row), pred in zip(df.iterrows(), predictions):
        true = row["label"]
        predicted = "consistent" if pred == 1 else "contradict"

        if true != predicted:
            errors.append({
                "id": row["id"],
                "book": row["book_name"],
                "caption": row["caption"],
                "true": true,
                "predicted": predicted,
                "content": row["content"][:300]
            })
    return errors
