import asyncio
import json

from BAScraper.BAScraper_async import PullPushAsync
from convokit import Corpus, download
from rich.progress import track


async def main(all_conversations: list):

    final_conv_counter = 0
    data_to_save = []
    for conv in track(all_conversations):
        if conv.meta["has_removed_comment"]:
            # Check that all messages in the conv have less than 500 characters (since we want conversations with short messages)

            valid_conv = True
            utterance_list = conv.get_chronological_utterance_list()
            if len(utterance_list) < 2 or len(utterance_list) >= 6:
                # Conversations with less than 2 messages are useless
                # Conversations with more than 6 messages appear in the Reddit dataset that we want to simulate
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
            statement = submission_title.replace("CMV:", "").strip()
            penultimate_utterance = utterance_list[-2]
            last_utterance = utterance_list[-1]
            split = conv["meta"].split()

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
        json.dump(
            data_to_save,
            f,
        )
    print(f"There are {final_conv_counter} final pairs.")


if __name__ == "__main__":
    ppa = PullPushAsync(log_stream_level="DEBUG", task_num=16)
    corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))
    all_conversations = list(corpus.iter_conversations())
    asyncio.run(main(all_conversations))
