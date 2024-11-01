from enum import Enum


class DemographicCharacteristic(Enum):
    """Demographic characteristic for agents.
        Based on:
            - https://github.com/yunshiuan/llm-agent-opinion-dynamics/blob/main/prompts/opinion_dynamics/Flache_2017/list_agent_descriptions.csv
            - 10.48550/arXiv.2310.05984 - Fig. 1
            - 10.5555/3618408.3619652 - Table 2
            - https://electionstudies.org/wp-content/uploads/2024/05/anes_specialstudy_2024ets_qnnaire_20240403.pdf
            - https://www.pewresearch.org/wp-content/uploads/sites/20/2024/07/2024-NPORS-Paper-Questionnaire.pdf 
            - https://yourmorals.org/ 
    """

    # NAME = "name" ; name is not a demographic characteristic but a required field for agents

    ETHNICITY = "ethnicity"

    BIOLOGICAL_SEX = "biological sex"

    GENDER_IDENTITY = "gender identity"

    NATIONALITY = "nationality"

    AGE = "age"

    MARITAL_STATUS = "marital status"

    EDUCATION = "education"

    OCCUPATION = "occupation"

    POLITICAL_LEANING = "political leaning"

    RELIGION = "religion or spiritual beliefs"

    SEXUAL_ORIENTATION = "sexual orientation"

    HEALTH_CONDITION = "health condition"

    INCOME = "total income of your family in the past 12 months"

    HOUSEHOLD_SIZE = "household size"

    NUMBER_OF_DEPENDENT = "number of dependent children or elderly family members"

    LIVING_QUARTERS = "living quarters"

    LANGUAGE_SPOKEN = "language spoken"

    CITY_OF_RESIDENCE = "city of residence"

    PRIMARY_MODE_OF_TRANSPORTATION = "primary mode of transportation"

    BACKGROUND = "background"