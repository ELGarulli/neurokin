import pandas as pd


def import_anipose_csv(path):
    """
    Imports csv from anipose, then converts it to a dlc-like multiindex dataframe with three levels:
    Level 1: scorer: This level exists to make it similar to dlc dataframes. It is set as 'scorer'
    Level 2: bodypart: This level describes the bodypart, (e.g. 'TailBase') and is read from the dataframe
    Level 3: axis: this level describes the axis (x,y,z) and is read from the dataframe

    Input: path to the csv
    Output: dataframe in correct shape for neurokin
    """
    df = pd.read_csv(path)
    dfs = []
    for column in df.columns:
        if column.endswith("_x") or column.endswith("_y") or column.endswith("_z"):
            bp = column[: column.index("_")]
            axis = column[column.index("_") + 1 :]
            df_col = convert_singleindex_to_multiindex_df(
                scorer="scorer", bodypart=bp, axis=axis, data=df[column]
            )
            dfs.append(df_col)
        else:
            df.drop(column, axis=1)

    df_preprocessed = pd.concat(dfs, axis=1)
    return df_preprocessed


def convert_singleindex_to_multiindex_df(scorer: str, bodypart, axis, data):
    pdindex = pd.MultiIndex.from_product(
        [[scorer], [bodypart], [axis]], names=["scorer", "bodyparts", "coords"]
    )

    # To create a multiindex dataframe with the data, we need to convert it to a np.array. Why? yeah... good question.
    data = data.to_numpy()

    df_multiindexed = pd.DataFrame(data=data, columns=pdindex)
    return df_multiindexed
