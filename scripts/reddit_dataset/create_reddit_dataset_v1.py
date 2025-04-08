"""Script to create a dataset from Reddit posts."""

import asyncio

from BAScraper.BAScraper_async import PullPushAsync
from convokit import Corpus, download
from rich.progress import track


async def main(all_conversations: list):

    final_conv_counter = 0
    for conv in track(all_conversations):
        if conv.meta["has_removed_comment"]:
            # Check that all messages in the conv have less than 500 characters
            # Because we know that OlMo2 is not pretrained on these convs.
            valid_conv = True
            utterance_list = conv.get_chronological_utterance_list()
            if len(utterance_list) < 3:
                continue

            for i, utt in enumerate(utterance_list):
                if (
                    len(utt.text) > 500
                    or utt.speaker.id != utterance_list[i % 2].speaker.id
                    or utt.speaker.id in ["[deleted]", "[removed]"]
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
            submission_text = full_submission[submission_id]["selftext"]
            submission_author = full_submission[submission_id]["author"]

            if (
                len(submission_text) > 500
                or submission_author != utterance_list[1].speaker.id
                or submission_text in ["[deleted]", "[removed]"]
            ):
                continue

            final_conv_counter += 1
            # Statement is submission_title but without the "CMV: " prefix if it is present
            statement = submission_title.replace("CMV: ", "")
            print("================")
            print("Statement: ", statement)
            print("Submission author: ", submission_author)
            print("Submission text: ", submission_text)

            for i, utt in enumerate(utterance_list):
                print(f"{utt.speaker.id}: {utt.text}")

            # Save the conv in a csv file stored in data/reddit_convs
            # The csv files columns are "User ID", "User Name", "Text", "Timestamp"

            # with open(
            #     f"data/reddit/cmv/conversations/{conv.meta['conversation_id']}.csv", "w"
            # ) as f:
            #     f.write("User ID,User Name,Text,Timestamp\n")
            #     for i, utt in enumerate(utterance_list):
            #         f.write(
            #             "{},{},{},{}\n".format(
            #                 i % 2, utt.meta["user_name"], utt.text, utt.timestamp
            #             )
            #         )
            break
    print(f"There are {final_conv_counter} final conversations.")


if __name__ == "__main__":
    ppa = PullPushAsync(log_stream_level="DEBUG", task_num=2)
    corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))
    all_conversations = list(corpus.iter_conversations())
    asyncio.run(main(all_conversations))
