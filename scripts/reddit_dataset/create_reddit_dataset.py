"""Script to create a dataset from Reddit posts."""  # TODO improve docstring

import asyncio
import csv
import json

from BAScraper.BAScraper_async import PullPushAsync
from convokit import Corpus, download
from rich.progress import track


async def main(all_conversations: list):

    final_conv_counter = 0
    submissionid_to_statement = {}
    for conv in track(all_conversations):
        if conv.meta["has_removed_comment"]:
            # Check that all messages in the conv have less than 500 characters
            # Because we know that OlMo2 is not pretrained on these convs.
            # TODO Add Reference to OlMo 2's pretraining dataset
            valid_conv = True
            utterance_list = conv.get_chronological_utterance_list()
            if len(utterance_list) < 6:
                continue

            for utt in utterance_list:
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
            full_comment = await ppa.fetch(mode="comments", ids=[first_comment_id])
            submission_id = full_comment[first_comment_id]["link_id"].split("_")[1]
            full_submission = await ppa.fetch(mode="submissions", ids=[submission_id])
            submission_title = full_submission[submission_id]["title"]

            if submission_title in ["[removed]", "[deleted]"]:
                continue

            final_conv_counter += 1
            # Statement is submission_title but without the "CMV: " prefix if it is present
            statement = submission_title.replace("CMV: ", "")

            # for utt in utterance_list:
            #     utt_text = utt.text.replace("&gt", ">")
            #     print(f"{utt.speaker.id}: {utt_text}")

            # Save the conv in a csv file stored in data/reddit_convs
            # The csv files columns are "User ID", "User Name", "Text", "Timestamp"

            submissionid_to_statement[submission_id] = statement

            with open(
                f"data/reddit/cmv/sub_{submission_id}-comment_{first_comment_id}.csv",
                "w",
                newline="",
            ) as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(["User ID", "User Name", "Text", "Timestamp"])
                for utt in utterance_list:
                    writer.writerow(
                        [
                            utt.speaker.id,
                            utt.speaker.id,
                            utt.text.replace("&gt;", "> "),
                            utt.timestamp,
                        ]
                    )
    with open("data/reddit/cmv/statements.json", "w") as f:
        json.dump(
            submissionid_to_statement,
            f,
        )
    print(f"There are {final_conv_counter} final conversations.")


if __name__ == "__main__":
    ppa = PullPushAsync(log_stream_level="DEBUG", task_num=16)
    corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))
    all_conversations = list(corpus.iter_conversations())
    asyncio.run(main(all_conversations))
