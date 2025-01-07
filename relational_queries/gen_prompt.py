"""
the system and user prompts for each relational queries

output: ./flat_prompts/{bird_id}.csv file with two fields:
    * sys_prompt, user_prompt

the system prompt is obtained from LOTUS: https://github.com/guestrin-lab/lotus
"""

sys_filter = (
        "The user will provide a claim and some relevant context.\n"
        "Your job is to determine whether the claim is true for the given context.\n"
        'You must answer with a single word, "True" or "False".'
        )

sys_map = (
        "The user will provide an instruction and some relevant context.\n"
        "Your job is to answer the user's instruction given the context."
        )

sys_topk = (
        "Your job is to to select and return the most relevant document to the user's question.\n"
        "Carefully read the user's question and the two documents provided below.\n"
        'Respond only with the label of the document such as "Document NUMBER".\n'
        "NUMBER must be either 1 or 2, depending on which document is most relevant.\n"
        'You must pick a number and cannot say things like "None" or "Neither"'
        )

import pandas as pd

def row_map(rel_type, rel_template, db_content_dict):
    """
    filter and map
    for each row, return a tuple: (system_prompt, user_prompt)
    """
    sys_input = sys_map if rel_type == "map" else sys_filter
    user_input = rel_template.format(**db_content_dict)
    return sys_input, user_input


def topk_process(rel_template, df):
    """
    pairwise compare, output the result
    """
    N = df.shape[0]
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            d1 = df.iloc[i]
            d2 = df.iloc[j]
            pairs.append((dict(d1), dict(d2)))

    sys_list = []
    user_list = []
    for d1, d2 in pairs:
        sys_list.append(sys_topk)
        user_prompt = f"Question: {rel_template}\n"
        for idx, doc in enumerate([d1, d2]):
            user_prompt += f"\nDocument {idx+1}:\n {doc}"
        user_list.append(user_prompt)

    return pd.DataFrame({"system": sys_list, "user": user_list})

if __name__ == "__main__":

    meta = pd.read_csv("meta_info.csv")
    for idx, row in meta.iterrows():
        bird_id = row['bird_id']
        rel_type = row['rel_type']
        rel_template = row['rel_template']
        rel_counts = row['rel_counts']

        df_path = f"dfs/{bird_id}.csv"
        df = pd.read_csv(df_path)

        if rel_type == 'topk':
            res_df = topk_process(rel_template, df)
        else:
            prompts = df.apply(lambda x: row_map(rel_type, rel_template, dict(x)), axis=1).values
            res_df = pd.DataFrame({"system": [p[0] for p in prompts], "user":[p[1] for p in prompts]})

        res_df.to_csv(f"flat_prompts/{bird_id}.csv", index=False)
        print(f"{bird_id} ok, {res_df.shape[0]}")

