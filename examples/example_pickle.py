"""Read a pickled conversation and display its metrics."""

# TODO: matplotlib

from llm_mediator_simulations.simulation.debate import Debate

data = Debate.unpickle("debate.pkl")


# Display the debate config
print("Debate configuration:")
print("--------------------")
print(data["config"])
print()

# Display the debaters config
print("Debaters configuration:")
print("----------------------")
for debater in data["debaters"]:
    print(debater)
print()


# Display the message data
print("Debate messages:")
print("----------------")
for message in data["messages"]:
    if message.authorId is None:
        print("Mediator -", message.text)
    else:
        debater = data["debaters"][message.authorId]
        print(
            debater.position,
            "-",
            message.metrics.perspective if message.metrics else "None",
            "-",
            message.text,
        )
    print()
