import wandb
import numpy as np

def get_col_from_df(df, col_name):
    return df[col_name].values

def df_to_np(df):
    return df.values

def get_time_diff(run_name):
    """Get the difference between first log and last log of a WANBB run"""
    api = wandb.Api()    
    run = api.run(run_name)
    df = run.history()["_timestamp"]
    arr = df_to_np(df)
    n = len(arr)
    for i in range(n-1):
        assert arr[i].item() < arr[i+1].item()
    print(arr[-1].item() - arr[0].item())

def get_nonan(arr, last=True):
    """Get last non nan by default (or first if last=False)"""
    
    indices = list(range(len(arr)-1, -1, -1)) if last else list(range(len(arr)))

    for i in indices: # range(len(arr)-1, -1, -1):
        if not np.isnan(arr[i]):
            return arr[i]

    return np.nan

def get_corresponding_element(
    df,
    col1_name,
    col1_value,
    col2_name, 
):
    """Get the corresponding element of col2_name for a given element of col1_name"""
    col1 = get_col_from_df(df, col1_name)
    col2 = get_col_from_df(df, col2_name)
    for i in range(len(col1)):
        if col1[i] == col1_value and not np.isnan(col2[i]):
            return col2[i]
    assert False, "No corresponding element found"

def get_first_element(
    df,
    col,
    last=False,
):
    col1 = get_col_from_df(df, "_step")
    col2 = get_col_from_df(df, col)

    cur_step = 1e30 if not last else -1e30
    cur_ans = None

    for i in range(len(col1)):
        if not last:
            if col1[i] < cur_step and not np.isnan(col2[i]):
                cur_step = col1[i]
                cur_ans = col2[i]
        else:
            if col1[i] > cur_step and not np.isnan(col2[i]):
                cur_step = col1[i]
                cur_ans = col2[i]

    assert cur_ans is not None
    return cur_ans

def get_longest_float(s, end_cutoff=None):
    ans = None
    if end_cutoff is None:
        end_cutoff = len(s)
    else:
        assert end_cutoff < 0, "Do -1 or -2 etc mate"

    for i in range(len(s)-1, -1, -1):
        try:
            ans = float(s[i:end_cutoff])
        except:
            pass
        else:
            ans = float(s[i:end_cutoff])
    assert ans is not None
    return ans

def get_threshold_zero(s, num=3):
    return float(s.split("_")[num])

def process_nan(tens):
    # turn nans into -1s
    assert isinstance(tens, np.ndarray)
    assert len(tens.shape) == 1, tens.shape
    tens[np.isnan(tens)] = -1
    tens[0] = tens.max()
    
    # turn -1s into the minimum value
    tens[np.where(tens == -1)] = 1000
    
    for i in range(1, len(tens)):
        tens[i] = min(tens[i], tens[i-1])
    return tens

    