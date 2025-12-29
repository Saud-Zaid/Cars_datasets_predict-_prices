from sklearn.model_selection import train_test_split

def random_split(df, *, target, test_size, seed, stratify):
    y = df[target]
    strat = y if stratify else None
    train, test = train_test_split(
    df, test_size=test_size, random_state=seed, stratify=strat
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)