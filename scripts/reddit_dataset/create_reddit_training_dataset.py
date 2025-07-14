import asyncio
import json
import os

from BAScraper.BAScraper_async import ArcticShiftAsync
from convokit import Corpus, download
from rich.progress import track


async def main(all_conversations: list):
    # In the dir data/reddit/cmv, there is alist of convs stored in csv files with the following name format:
    # sub_{submission_id}-comment_{first_comment_id}.csv
    # Create a hashtable storing f"{submision_id}-{first_comment_id}"
    csga_cmv_simulation_conv_ids = set()
    # list all the csv files in the dir data/reddit/cmv
    for filename in os.listdir("data/reddit/cmv"):
        if filename.endswith(".csv"):
            # Get the submission_id and comment_id from the filename
            submission_id = filename.split("-")[0].split("_")[1]
            comment_id = filename.split("-")[1].split(".")[0].split("_")[1]
            csga_cmv_simulation_conv_ids.add(f"{submission_id}-{comment_id}")

    assert (
        len(csga_cmv_simulation_conv_ids) == 177
    ), "Error: there are not 177 convs in the data/reddit/cmv folder"

    final_conv_counter = 0
    csga_cmv_simulation_found_convs = 0
    data_to_save = []
    for conv in track(all_conversations):
        if conv.meta["has_removed_comment"]:
            # Check that all messages in the conv have less than 500 characters (since we want conversations with short messages)

            valid_conv = True
            utterance_list = conv.get_chronological_utterance_list()
            if len(utterance_list) < 2:
                # Conversations with less than 2 messages are useless
                continue

            for utt in utterance_list[-2:]:
                # We only check the last two utterances, since the first ones are not relevant
                if (
                    # (i < 6 and len(utt.text) > 500)
                    len(utt.text) > 500
                    or utt.text in ["[deleted]", "[removed]"]
                ):
                    valid_conv = False
                    break
            if not valid_conv:
                continue

            first_comment_id = utterance_list[0].id
            full_comment = await asa.fetch(
                mode="comments_id_lookup", ids=[first_comment_id]
            )
            submission_id = full_comment[first_comment_id]["link_id"].split("_")[1]

            if f"{submission_id}-{first_comment_id}" in csga_cmv_simulation_conv_ids:
                # Check whether the conv id is in the csga_cmv_simulation_conv_ids
                # If it is, then we skip this conversation since we want no overlap
                # between the dataset being created right now and the csga_cmv_simulation dataset
                csga_cmv_simulation_found_convs += 1
                continue

            full_submission = await asa.fetch(
                mode="submissions_id_lookup", ids=[submission_id]
            )
            submission_title = full_submission[submission_id]["title"]

            if submission_title in ["[removed]", "[deleted]"]:
                continue

            final_conv_counter += 1
            # Statement is submission_title but without the "CMV: " prefix if it is present
            statement = submission_title.replace("CMV:", "").strip()
            penultimate_utterance = utterance_list[-2]
            last_utterance = utterance_list[-1]
            split = conv.meta["split"]

            record = {
                "statement": statement,
                "submission_id": submission_id,
                "penultimate_utterance": {
                    "id": penultimate_utterance.id,
                    "userid": penultimate_utterance.speaker.id,
                    "text": penultimate_utterance.text.replace("&gt;", "> "),
                    "timestamp": penultimate_utterance.timestamp,
                },
                "last_utterance": {
                    "id": last_utterance.id,
                    "userid": last_utterance.speaker.id,
                    "text": last_utterance.text.replace("&gt;", "> "),
                    "timestamp": last_utterance.timestamp,
                },
                "split": split,
            }

            data_to_save.append(record)

    with open("data/reddit/cmv/cga_cmv_pairs_before_derailment.jsonl", "w") as f:
        for record in data_to_save:
            f.write(json.dumps(record) + "\n")
    print(f"There are {final_conv_counter} final pairs.")
    assert csga_cmv_simulation_found_convs == 177


if __name__ == "__main__":
    # ppa = PullPushAsync(log_stream_level="DEBUG", task_num=1) # PullPush Scheduled Maintenance
    # The Pullpush API will be temporarily unavailable until mid-May due to essential hardware upgrades and a full reindexing process.

    asa = ArcticShiftAsync(log_stream_level="DEBUG", task_num=16)

    corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))
    all_conversations = list(corpus.iter_conversations())
    asyncio.run(main(all_conversations))
