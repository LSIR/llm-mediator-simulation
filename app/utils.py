from matplotlib import cm, pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import streamlit as st
from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.scales import Likert3Level, Likert7AgreementLevel
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.simulation.debater.config import DebaterConfig, TopicOpinion


import random

MODELS = ["deepseek-r1:8b", "deepseek-r1:1.5b", "llama3.2", "mistral", "mistral-nemo"]

UNISEXNAMES = ["Alex", "Riley", "Jordan", "Parker", "Sawyer", "Taylor", "Casey", "Avery", "Jamie", "Quinn"]


def get_random_debater_profile(agent_num):
    name = UNISEXNAMES[agent_num]
    # avatar = st.sidebar.text_input(f"Agent {agent_num} Avatar URL", value=f"https://via.placeholder.com/50?text=A{agent_num}")
    # return {"name": name} # , "avatar": avatar}

    personality = Personality(
        demographic_profile={DemographicCharacteristic.ETHNICITY: random.choice(["White", "Black", "Asian", "Hispanic"]),
                             DemographicCharacteristic.BIOLOGICAL_SEX: random.choice(["Male", "Female"]),
                             DemographicCharacteristic.NATIONALITY: random.choice(["American"]),
                             DemographicCharacteristic.AGE: str(random.randint(18, 80)),
                             DemographicCharacteristic.MARITAL_STATUS: random.choice(["Single", "Married", "Divorced", "Widowed"]),
                             DemographicCharacteristic.EDUCATION: random.choice(["High School", "College", "Graduate School"]),
                             DemographicCharacteristic.OCCUPATION: random.choice(["Engineer", "Doctor", "Teacher", "Artist", "Unemployed", "Student"]),
                             DemographicCharacteristic.POLITICAL_LEANING: random.choice(["Democrat", "Republican", "Independent"]),
                             DemographicCharacteristic.RELIGION: random.choice(["Christian", "Muslim", "Jewish", "Atheist"]),
                             DemographicCharacteristic.SEXUAL_ORIENTATION: random.choice(["Heterosexual", "Homosexual", "Bisexual"]),
                             DemographicCharacteristic.HEALTH_CONDITION: random.choice(["Disabled", "Non-disabled"]),
                             DemographicCharacteristic.INCOME: str(random.randint(30000, 100000)),
                             DemographicCharacteristic.HOUSEHOLD_SIZE: str(random.randint(1, 5)),
                             DemographicCharacteristic.NUMBER_OF_DEPENDENT: str(random.randint(0, 3)),
                             DemographicCharacteristic.LIVING_QUARTERS: random.choice(["House", "Apartment", "Condo"]),
                             DemographicCharacteristic.CITY_OF_RESIDENCE: random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]),
                             DemographicCharacteristic.PRIMARY_MODE_OF_TRANSPORTATION: random.choice(["Car", "Public Transit", "Bicycle", "Walking"]),
                             }
    )

    traits = {}
    random_traits = random.sample(list(PersonalityTrait), random.randint(1, 5))
    for trait in random_traits:
        traits[trait] = random.choice([Likert3Level.LOW, Likert3Level.AVERAGE, Likert3Level.HIGH])

    personality.traits = traits
    return DebaterConfig(name=name,
                         topic_opinion=TopicOpinion(agreement=random.choice(list(Likert7AgreementLevel))),
                         personality = personality)


def get_predefined_debater_profile(agent_num):
    name = UNISEXNAMES[agent_num]
    if agent_num == 0:
        personality = Personality(
            demographic_profile={DemographicCharacteristic.ETHNICITY: "White",
                                DemographicCharacteristic.BIOLOGICAL_SEX: "Male",
                                DemographicCharacteristic.NATIONALITY: "American",
                                DemographicCharacteristic.AGE: "70",
                                DemographicCharacteristic.MARITAL_STATUS: "Married",
                                DemographicCharacteristic.EDUCATION: "High School",
                                DemographicCharacteristic.OCCUPATION: "Worker",
                                DemographicCharacteristic.POLITICAL_LEANING: "Republican",
                                DemographicCharacteristic.RELIGION: "Christian",
                                DemographicCharacteristic.SEXUAL_ORIENTATION: "Heterosexual",
                                DemographicCharacteristic.HEALTH_CONDITION: "Non-disabled",
                                DemographicCharacteristic.INCOME: "US$ 50,000",
                                DemographicCharacteristic.HOUSEHOLD_SIZE: "5",
                                DemographicCharacteristic.NUMBER_OF_DEPENDENT: "3",
                                DemographicCharacteristic.LIVING_QUARTERS: "House",
                                DemographicCharacteristic.CITY_OF_RESIDENCE: "Midland, Texas",
                                DemographicCharacteristic.PRIMARY_MODE_OF_TRANSPORTATION: "Car",
                                },
            traits={PersonalityTrait.AGREEABLENESS: Likert3Level.LOW,
                    PersonalityTrait.CONSCIENTIOUSNESS: Likert3Level.LOW,
                    PersonalityTrait.EXTRAVERSION: Likert3Level.LOW,
                    PersonalityTrait.NEUROTICISM: Likert3Level.LOW,
                    PersonalityTrait.OPENNESS: Likert3Level.LOW}
        )
        return DebaterConfig(name=name, 
                             topic_opinion=TopicOpinion(agreement=Likert7AgreementLevel.STRONGLY_AGREE), 
                             personality=personality)
    
    if agent_num == 1:
        personality = Personality(
            demographic_profile={DemographicCharacteristic.ETHNICITY: "Black",
                                 DemographicCharacteristic.BIOLOGICAL_SEX: "Female",
                                 DemographicCharacteristic.NATIONALITY: "American",
                                 DemographicCharacteristic.AGE: "28",
                                    DemographicCharacteristic.MARITAL_STATUS: "Single",
                                    DemographicCharacteristic.EDUCATION: "Ph.D.",
                                    DemographicCharacteristic.OCCUPATION: "Social Scientist",
                                    DemographicCharacteristic.POLITICAL_LEANING: "Democrat",
                                    DemographicCharacteristic.RELIGION: "Muslim",
                                    DemographicCharacteristic.SEXUAL_ORIENTATION: "Homosexual",
                                    DemographicCharacteristic.HEALTH_CONDITION: "Non-disabled",
                                    DemographicCharacteristic.INCOME: "US$ 180,000",
                                    DemographicCharacteristic.HOUSEHOLD_SIZE: "1",
                                    DemographicCharacteristic.NUMBER_OF_DEPENDENT: "0",
                                    DemographicCharacteristic.LIVING_QUARTERS: "Apartment",
                                    DemographicCharacteristic.CITY_OF_RESIDENCE: "San Francisco, California",
                                    DemographicCharacteristic.PRIMARY_MODE_OF_TRANSPORTATION: "Bike",
                                    },
            traits={PersonalityTrait.AGREEABLENESS: Likert3Level.LOW,
                    PersonalityTrait.CONSCIENTIOUSNESS: Likert3Level.HIGH,
                    PersonalityTrait.EXTRAVERSION: Likert3Level.HIGH,
                    PersonalityTrait.NEUROTICISM: Likert3Level.HIGH,
                    PersonalityTrait.OPENNESS: Likert3Level.LOW}
        )
        return DebaterConfig(name=name,
                                topic_opinion=TopicOpinion(agreement=Likert7AgreementLevel.STRONGLY_DISAGREE),
                                personality=personality)



    




                             

SEED = 42


def streamlit_plot_metrics(debate, metric_to_plot):
    fig, ax = plt.subplots()
    if debate:
        values = []
        debater_intervention_num = 0
        for intervention in debate.interventions:
            if intervention.metrics:
                debater_intervention_num += 1
                values.append(intervention.metrics.argument_qualities[metric_to_plot].value + 1)
        
        cmap = cm.get_cmap('RdYlGn')  # Red (low) to Green (high)
        norm = mcolors.Normalize(vmin=1, vmax=5)  # Normalize between 1 and 5
        

        x_values = np.arange(1, len(values) + 1)
        colors = [cmap(norm(v)) for v in values]
        
        for i in range(len(values) - 1):
            x_segment = [x_values[i], x_values[i+1]]
            y_segment = [values[i], values[i+1]]
            lc = np.linspace(norm(values[i]), norm(values[i+1]), 10)
            for j in range(len(lc) - 1):
                ax.plot(
                    np.linspace(x_segment[0], x_segment[1], 10)[j:j+2],
                    np.linspace(y_segment[0], y_segment[1], 10)[j:j+2],
                    color=cmap(lc[j]), linewidth=2
                )
            ax.scatter(x_segment, y_segment, color=[colors[i], colors[i+1]], edgecolors='k')
            
        
    ax.set_xlabel("Intervention")
    ax.set_ylabel("Metric Score")
    # x scale steps of 1
    ax.set_xlim(0, max(debater_intervention_num + 1 , 10))
    ax.set_xticks(range(1, max(debater_intervention_num + 1 , 10)))
    # y scale between 0 and 5
    ax.set_ylim(0.8, 5.5)
    ax.set_yticks(range(1, 6))
    ax.set_title("Evolution of Argument Quality Metrics")
    ax.legend([metric_to_plot.value[0]])
    st.pyplot(fig)

def flip_metric(metric):
    """Workaroun to probel of checkbox not updating session state..."""
    if st.session_state[f"check_{metric.value[0]}"]:
        st.session_state.metrics[metric] = True
    else:
        st.session_state.metrics[metric] = False    

def flip_debate_type(debate_type):
    """Workaroun to probel of checkbox not updating session state..."""
    if st.session_state[debate_type]:
        st.session_state[debate_type] = False
    else:
        st.session_state[debate_type] = True