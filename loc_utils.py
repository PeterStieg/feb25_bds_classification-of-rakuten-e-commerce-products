import numpy as np
import pandas as pd
import deepl
import os
deepl_file = 'language_analysis/df_localization.csv'
deepl_output_file = 'language_analysis/deepL_result.csv'
ignore_language_codes = ['fr', 'ja', 'el', 'ko', 'mul', 'zh', 'la', 'vo', '??', 'ru', 'el',  'und']
auth_key = os.environ.get("DEEPL_API_KEY", None) 
translator = deepl.Translator(auth_key) # check if API key exists in .env file


def import_clean_data():
    X_test = pd.read_csv("X_test.csv", index_col=0)
    X_train = pd.read_csv("X_train.csv", index_col=0)
    y_train = pd.read_csv("y_train.csv", index_col=0)
    df = pd.merge(X_train, y_train, left_index=True, right_index=True)
    df = text_utils.text_pre_processing(df)
    df = df[['productid', 'imageid', 'prdtypecode', 'bool_description','merged_text']]
    df_lang = pd.read_csv('language_analysis/df_langdetect.csv', index_col=0)
    df_lang['lang'] = df_lang['gemini_lang'].fillna('fr')
    df_lang.drop(['merged_text', 'merged_langdetect', 'gemini_lang', 'imageid', 'prdtypecode', 'bool_description'], inplace=True, axis = 1)
    df_lang = pd.merge(df, df_lang, on = ['productid'], how='left')
    df_lang.to_csv('language_analysis/df_lang.csv')
    return df_lang
def deepl_translation(prompt, target_lang = 'fr'):
    result = translator.translate_text(prompt, target_lang=target_lang)
    return result

def safe_create_column(column, data_frame):
    '''only creates a new empty column if it does not already exist'''
    if column not in data_frame.columns:
        data_frame.loc[:, column] = np.nan  # Use loc to avoid SettingWithCopyWarning
    return data_frame

def apply_translation_conditionally(df):
    mask = (
        ~df['lang'].isin(ignore_language_codes) &
        # removing french, language codes which do not use latin script as well as nonsense language codes
        # The fact that the LLM failed might indicate that there is something off with these strings
        (df['deepL_translation'].isna())
    )
    df.loc[mask, 'deepL_translation'] = df.loc[mask, 'merged_text'].apply(deepl_translation)
    return df

def apply_translation_conditionally_in_chunks(df, chunk_size=200, output_file=deepl_output_file):
    total_rows = len(df)
    rows_left = total_rows
    # Loop over the DataFrame in chunks
    for start_row in range(0, total_rows, chunk_size):
        # Define the chunk
        end_row = min(start_row + chunk_size, total_rows)
        df_chunk = df.iloc[start_row:end_row]
        
        # Apply the translation function on this chunk
        df_chunk = apply_translation_conditionally(df_chunk)
        
        # Update the original DataFrame with changes from the chunk
        df.update(df_chunk)
        
        # Write the processed chunk to CSV (append mode)
        df_chunk.to_csv(output_file, mode='a', index=True)
        
        # After the first chunk, subsequent chunks should not include the header
        rows_left -= chunk_size
        print(f'{rows_left} rows left to run')
    return df


# TODO add multithreading 
# 
# from concurrent.futures import ThreadPoolExecutor

# def apply_translation_parallel(df, max_workers=10):
#     mask = (
#         (df['lang'] == 'en') &
#         (df['prdtypecode'] == 10) &
#         (df['deepL_translation'].isna())
#     )

#     indices = df[mask].index
#     texts = df.loc[indices, 'merged_text']

#     def translate_row(i, text):
#         return (i, deepl_translation(text))

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(translate_row, i, text) for i, text in zip(indices, texts)]
#         for future in futures:
#             i, translation = future.result()
#             df.at[i, 'deepL_translation'] = translation