def summary_prompt(
    messages: list[str], summary: str, utterance: str = "messages", ignore: bool = False
) -> str:
    msg_sep = "\n\n"
    if not messages:
        return ""

    prompt = ""

    if not ignore:
        prompt += f"""Here is a summary of the conversation so far:
{summary}\n\n"""  # TODO Add personalized summary... "According to you, here is a summary..."

    prompt += f"""Here are the last {utterance}s:
{msg_sep.join(messages)}
"""

    return prompt
