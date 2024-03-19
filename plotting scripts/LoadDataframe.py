from tbparse import SummaryReader
import pandas as pd

def load_dataframe(log_dir):

    reader = SummaryReader(log_dir)

    df = reader.tensors

    # Rename
    df = df.rename(columns={'step': 'Epoch'})

    df = df.set_index(['Epoch'])

    # For each tag - there must be a column
    tags = df.loc[:, "tag"].unique()

    data = {}

    for tag in tags:
        mask = df["tag"] == tag
        
        df_tmp = df.loc[mask]
        
        new_tag = tag.replace("_", " ")

        data[new_tag] = df_tmp.value 

    df = pd.DataFrame(data)

    return df