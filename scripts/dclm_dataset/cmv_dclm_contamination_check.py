"""Script to check if the CMV Reddit users and submissions are contaminated with dclm-baseline documents.
Important:
1) check that your HF Cache location does not create deadlock issues
-> in .bashrc, `export HF_HOME="/mnt/datasets/dclm-baseline-2"`
and don't forget to revert back.
"""

import functools
import json
import multiprocessing
import os

import pandas as pd
from datasets import load_dataset
from rich.progress import track


def check_dclm_contamination_cmv_submissions():
    # Check if the URLs of some CMV convs whose ids are stored in data/reddit/cmv/statements.json
    # are in the data/reddit/cmv/dclm_contamination/cmv_reddit_urls.json
    with open(
        "data/reddit/cmv/statements.json",
        "r",
    ) as f:
        # The file contains a dictionary of submission_ids and statements
        submissionid_to_statement = json.load(f)
    with open(
        "data/reddit/cmv/dclm_contamination/cmv_reddit_urls.json",
        "r",
    ) as f:
        # The file contains a dictionary of shard_index and urls
        cmv_reddit_urls = json.load(f)

    contamination_counter = 0
    comparison_made = 0
    for submission_id in track(submissionid_to_statement.keys()):
        for shard_index, url in cmv_reddit_urls.items():
            comparison_made += 1
            if submission_id in url:
                contamination_counter += 1
                print(
                    f"Found a CMV Reddit submission id: {submission_id} in the URL {url} of dclm-baseline shard {shard_index}"
                )

    return contamination_counter, comparison_made


def process_url(url_item, ds, num_shards):
    local_url_to_doc = {}
    key, url = url_item
    shard_index, _ = key.split("_")
    ds_shard = ds.shard(num_shards=num_shards, index=int(shard_index))

    for example in ds_shard:
        if example["url"] == url:

            local_url_to_doc[key] = example["text"].replace("&gt;", "> ")
    return local_url_to_doc


def collect_document_texts_of_contaminated_redditcmv_users():
    with open(
        "data/reddit/cmv/dclm_contamination/cmv_reddit_users_urls.json",
        "r",
    ) as f:
        # The file contains a dictionary of shardIndex_idInTheShard and Reddit user urls
        cmv_reddit_users_urls = json.load(
            f
        )  # 83 keys but 71 unique urls (so duplicate URLs)

    ds = load_dataset(
        "mlfoundations/dclm-baseline-1.0-parquet",
        split="train",
        streaming=True,
        columns=["text", "url"],
    )
    num_shards = ds.num_shards  # type: ignore

    process_url_local = functools.partial(process_url, ds=ds, num_shards=num_shards)

    global_url_to_doc = {}

    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(process_url_local, cmv_reddit_users_urls.items())

    # Combine the results from all processes
    for result in results:
        global_url_to_doc.update(result)

    return global_url_to_doc


def check_dclm_contamination_cmv_users(url_to_doc):
    # join dict at "data/reddit/cmv/dclm_contamination/cmv_reddit_users_urls.json" (dict of key: url) and url_to_doc (dict of key: doc)
    # into a single dict of key: (url, doc)
    with open(
        "data/reddit/cmv/dclm_contamination/cmv_reddit_users_urls.json",
        "r",
    ) as f:
        # The file contains a dictionary of shard_index and urls
        cmv_reddit_users_urls = json.load(f)

    merged_dict = {}
    for key, url in cmv_reddit_users_urls.items():
        assert key in url_to_doc
        merged_dict[key] = (url, url_to_doc[key])

    user_contamination_counter = 0

    # iterate over all csv files in data/reddit/cmv
    for file in track(os.listdir("data/reddit/cmv")):
        if file.endswith(".csv"):
            path = os.path.join("data/reddit/cmv", file)
            df = pd.read_csv(path)
            # iterate over all rows in the dataframe
            for _, row in df.iterrows():
                user_id = row["User ID"]
                text = row["Text"]

                for key, (url, doc) in merged_dict.items():
                    if user_id in url:
                        # Check if the text is in the doc
                        if (
                            text in doc
                        ):  # Searching for the full text in the doc is not very subtil but let's assume that it's reasonable
                            user_contamination_counter += 1
                            print(
                                f"Found a text written by CMV Reddit user id: {user_id} in the URL {url} that is both in the filtered CGA-CSV and in the dclm-baseline document {doc}"
                            )
    return user_contamination_counter


if __name__ == "__main__":
    submission_contaminations, comparisons = check_dclm_contamination_cmv_submissions()
    print(
        f"Found {submission_contaminations} CMV Reddit submission ids in the dclm-baseline URLs after {comparisons} comparisons."
    )
    assert submission_contaminations == 0
    # output -> Found 0 CMV Reddit submission ids in the dclm-baseline URLs after 302670 comparisons.

    url_to_doc = collect_document_texts_of_contaminated_redditcmv_users()

    user_contamination_counter = check_dclm_contamination_cmv_users(url_to_doc)
    print(
        f"Found {user_contamination_counter} CMV Reddit users-written texts in the dclm-baseline documents."
    )
    assert user_contamination_counter == 0
    # output -> Found 0 CMV Reddit users-written texts in the dclm-baseline documents.
