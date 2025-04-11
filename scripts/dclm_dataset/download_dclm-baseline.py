"""Script to download the DCLM Baseline dataset in streaming mode from Hugging Face and extract CMV Reddit URLs and users
Important, login with Authenticate with Hugging Face to avoid huggingface_hub.errors.HfHubHTTPError: 429 Client Error: Too Many Requests
`huggingface-cli login`
"""

import json
import multiprocessing
import re

from datasets import load_dataset

repo_id = "mlfoundations/dclm-baseline-1.0-parquet"

# Load the user_ids
with open(
    "data/reddit/cmv/user_ids.json",
    "r",
) as f:
    # The file contains a list of user_ids
    user_ids = json.load(f)

user_ids = set(user_ids)

print("Number of user_ids: ", len(user_ids))

ds = load_dataset(
    "mlfoundations/dclm-baseline-1.0-parquet",
    split="train",
    streaming=True,
    columns=["url"],
)

NUM_SHARDS = ds.num_shards  # type: ignore

print("ds loaded, number of shards = ", NUM_SHARDS)
print(ds)


global_cmv_reddit_urls = {}
global_cmv_reddit_users_urls = {}

changemyview_regex = re.compile(r"reddit.com/r/changemyview/")
user_regex = re.compile(r"reddit.com/(?:user|u)/([^/?#]+)")


def process_shard(shard_index):
    ds_shard = ds.shard(num_shards=NUM_SHARDS, index=shard_index)  # type: ignore
    print("Starting to process shard ", shard_index)

    local_cmv_reddit_urls = {}
    local_cmv_reddit_users_urls = {}

    for i, example in enumerate(ds_shard):
        url = example["url"]

        if not url:
            print("Record has a non-standard URL field: ", example["url"])
            continue

        # Check if the URL is a CMV Reddit URL
        cmv_match = changemyview_regex.search(url)
        if cmv_match:
            local_cmv_reddit_urls[f"{shard_index}_{i}"] = url
            print("Found a CMV Reddit URL: ", url, " in shard ", shard_index)

        user_match = user_regex.search(url)
        if user_match:
            user_id = user_match.group(1)
            # Check if the user_id is in the list of user_ids
            if user_id in user_ids:
                local_cmv_reddit_users_urls[f"{shard_index}_{i}"] = url
                print(
                    "Found a CMV Reddit URL for a user: ",
                    url,
                    " in shard ",
                    shard_index,
                )

    return (
        local_cmv_reddit_urls,
        local_cmv_reddit_users_urls,
    )  # Return the dictionaries for this shard


# Setup multiprocessing pool
with multiprocessing.Pool(
    processes=8  # 8
) as pool:  # ~30 sec for 128 shards with 8 workers and 11 CMV URLs and 0 user. -> ~2h full dataset and ~2400 CMV Reddit convs
    process_num = NUM_SHARDS
    results = pool.map(process_shard, range(process_num))
# Combine results from all shards
for local_cmv_reddit_urls, local_cmv_reddit_users_urls in results:
    global_cmv_reddit_urls.update(local_cmv_reddit_urls)
    global_cmv_reddit_users_urls.update(local_cmv_reddit_users_urls)


print("Total number of CMV Reddit URLs: ", len(global_cmv_reddit_urls))
print("Total number of CMV Reddit URLs for users: ", len(global_cmv_reddit_users_urls))

with open("cmv_reddit_urls.json", "w") as f:
    json.dump(global_cmv_reddit_urls, f)

with open("cmv_reddit_users_urls.json", "w") as f:
    json.dump(global_cmv_reddit_users_urls, f)
