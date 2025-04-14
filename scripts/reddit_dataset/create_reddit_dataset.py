"""Script to create a dataset from the Conversation Gone Awry ChangeMyView Reddit dataset.
The datastaset will be useful to evaluate the capability of LLMs to synthesize the truncated part of the discussion

######### OlMo2 https://arxiv.org/pdf/2501.00656 Section 2.1.1.
"We combine data from DCLM (Li et al., 2024) and Dolma 1.7 (Soldaini et al., 2024). From DCLM, we use the “baseline 1.0” mix (3.71T tokens)"
(https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0).
From Dolma, we use [non-Reddit data...]

## DCLM
https://arxiv.org/pdf/2406.11794
Section 3.1 and Appendix E: DCLM-Pool (240 T tokens from all 5.1M Common Crawl WARC dumps from 2008 to 2022 (inclusive)).
# Section 3.1. "DCLM-POOL is an unfiltered web-text corpus comprised of all Common Crawl"
# Appendix E: "DCLM-POOL was collected by taking all 5.1M Common Crawl WARC dumps from 2008
to 2022 (inclusive) and extracting text from the html using the resiliparse framework."
# Section 4.2. "Text extraction is a common early processing step that pulls content from raw HTML.
To understand the effect of this step, we compare three text extraction approaches:
resiliparse, trafilatura (used by RefinedWeb), and the Common Crawl-provided
WET files that contain pre-extracted text."
-> So we understand that they do not unzip archive files that might have been collected and stored in Common Crawl WARC files.
And to our knowledge, as of today April 2025, the ConvoKit CGA CMV corpus is only accessible on 1) Reddit.com / Pushshift API and 2) on
the ConvKit web server as a compressed file at http://zissou.infosci.cornell.edu/convokit/datasets/conversations-gone-awry-cmv-corpus/conversations-gone-awry-cmv-corpus.zip
# Section 4.2. "We then apply RefinedWeb’s heuristic quality
filters to each of the text extractions"
# RefineWeb's Heuristics are there: https://github.com/mlfoundations/dclm/blob/e1d11ad9e9cee3b3c6c3c189117714b4280f3782/baselines/baselines_configs/dclm_baseline_refinedweb.yaml
"# Pipeline for RW filters used in DCLM-Baseline. The only difference is we remove the "high-quality" url filter (e.g. for wikipedia, github, etc.)" (See Appendix G.1.3. of https://arxiv.org/pdf/2306.01116)
-> So, even if RefineWeb's "high-quality" url filter removed Reddit.com data, DCLM-Baseline does not remove Reddit.com data from Common Crawl.
# Section 4.4. "For DCLM-BASELINE and the remaining experiments, we use fastText OH-2.5 + ELI5"
-> It means that Olmo2 is not pretrained on Bad conversations from Reddit. (+ Bad language has been filtered out using RefineWeb's Heuristics)

In Olmo 1, they implicitly assumed that no Reddit data are extracted from Common Crawl since they explicitly added Reddit Pushshift data in the pretraining in addition to common crawl.
But they never removed Reddit data from Common Crawl... See my question to the authors https://github.com/allenai/OLMo/issues/824
Side note: OlMo 2's pretraining dataset does not add Reddit data to Common Crawl data. Maybe they realized after OlMo1 that Reddit data were already in Common Crawl data?
It means that OlMo 2 might be pretrained on Reddit data (Including CMV)
We checked the URLs in DCML-Baseline's 2.95B documents ; 6.6 TiB.
https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
Is CMV in these URLs? If no, then we're safe.
Yes... https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet/viewer?sql=SELECT+*%0AFROM+train%0AWHERE+contains%28url%2C+%27www.reddit.com%2Fr%2Fchangemyview%27%29%0ALIMIT+100%3B&views%5B%5D=train
But not that many samples. -> We checked and there is 0 URL from the Reddit submissions of our curated CGA CMV dataset and
even if 83 users present in our curated CGA CMV dataset have their "user URL" present in dclm-baseline-1.0, we did not find that the correspondind documents contained individual utterences present in our curated CGA CMV dataset.

Besides, Reddit.com is in the list of URLs so OlMo2 learned to generate Reddit text.
https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet/viewer/default/train?f%5Burl%5D%5Bmin%5D=13&f%5Burl%5D%5Bmax%5D=250&f%5Burl%5D%5Btransform%5D=length&sql=SELECT+*%0AFROM+train%0AWHERE+contains%28url%2C+%27reddit.com%27%29%0ALIMIT+100%3B&views%5B%5D=train
(OlMo2 is not post-trained on Reddit data)
"""

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
            # Check that all messages in the conv have less than 500 characters (since we want conversations with short messages)

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
