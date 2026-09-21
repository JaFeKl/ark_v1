import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.experiment import DatasetItem, Evaluation
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from ark_v1.ark_v1 import ARK_V1
from ark_v1.data_models.agent_models import (
    FinalAnswerYesNo,
)

env_path = Path(__file__).parent.parent / ".env"

load_dotenv(dotenv_path=env_path)

# Initialize Langfuse CallbackHandler for Langchain
langfuse = get_client()
langfuse_handler = CallbackHandler()


# ------------------------------------------------
# Experiment set-up
# ------------------------------------------------

# Experiment metadata
experiment_metadata = {
    "model": "qwen3.5:9b",
    "provider": "LiteLLM",
    "temperature": 0.1,
    "seed": 42,
}

LITE_LLM_BASE_URL = os.environ.get("LITELLM_BASE_URL")
LITE_LLM_VIRTUAL_KEY = os.environ.get("LITELLM_VIRTUAL_KEY")


llm = ChatOpenAI(
    model=experiment_metadata["model"],
    base_url=LITE_LLM_BASE_URL,
    api_key=LITE_LLM_VIRTUAL_KEY,
    temperature=experiment_metadata["temperature"],
    seed=experiment_metadata["seed"],
)

agent = ARK_V1()
agent.llm = llm
agent.load_configuration(config={})

# # load example data
# with open(
#     Path(__file__).parent / "example_data.json",
#     "r",
# ) as file:
#     example_data = json.load(file)

# agent.load_graph_data(example_data["graph"])
# agent.set_initial_state(question=example_data["question"])
# final_state = agent.run()

# print(
#     f"Final answer: {final_state.get('finalAnswer').get('answer', 'No answer found')}"
# )


# Define the task function we pass to the experiment runner method
# def my_task(*, item, **kwargs):

#     # initialize

#     # print(question)
#     # print(response["messages"][1].content)
#     return "The capital of France is Paris."


# # Fetch dataset and run experiment
# dataset = langfuse.get_dataset(dataset_name)

# result = dataset.run_experiment(
#     name="test_experiment",
#     description="A simple test experiment for LangGraph agent",
#     task=my_task,
#     metadata={"model": "gpt-4o"},
# )


# Define the task function we pass to the experiment runner method
async def my_task(*, item: DatasetItem, **kwargs):
    # Initialize the agent with the input data
    agent.load_graph_data(item.metadata["graph"])
    agent.set_initial_state(question=item.input)
    output = await agent.run(langfuse_callback_handler=langfuse_handler)
    final_answer = output.get("finalAnswer", {}).get("answer", "No answer found")
    if final_answer is None:
        final_answer = "Unanswered"
    return final_answer


# ------------------------------------------------
# Evaluation set-up
# ------------------------------------------------

# there are different type of evaluators on different levels.


# Item-level evaluator
def accuracy_evaluator(*, input, output, expected_output, metadata, **kwargs):
    if expected_output == output:  # comparing two boolean in this example
        return Evaluation(name="accuracy", value=1.0, comment="Correct answer found")
    return Evaluation(name="accuracy", value=0.0, comment="Incorrect answer")


# Run-level evaluator
def average_accuracy(*, item_results, **kwargs):
    """Calculate average accuracy across all items"""
    accuracies = [
        eval.value
        for result in item_results
        for eval in result.evaluations
        if eval.name == "accuracy"
    ]
    if not accuracies:
        return Evaluation(name="avg_accuracy", value=None)
    avg = sum(accuracies) / len(accuracies)
    return Evaluation(
        name="avg_accuracy", value=avg, comment=f"Average accuracy: {avg:.2%}"
    )


# ------------------------------------------------
# Experiment execution
# ------------------------------------------------

# Fetch dataset and run experiment
dataset = langfuse.get_dataset("CR-LT-KGQA test")

result = dataset.run_experiment(
    name="my_demo_experiment",
    description="A simple test experiment for the ARK V1 agent",
    task=my_task,
    metadata=experiment_metadata,
    evaluators=[accuracy_evaluator],
    run_evaluators=[average_accuracy],
)

# Flush the langfuse client to ensure all data is sent to the server at the end of the experiment run
langfuse.flush()
print("Experiment run completed. Run ID:", result.dataset_run_id)


# ------------------------------------------------
# Experiment results retrieval
# ------------------------------------------------
# experiment_id = "6ed6346c-8b75-43b4-b1b1-6e40fd0b0c20"

# experiment_results = langfuse.api.experiments.list(
#     from_start_time=dataset.created_at, dataset_id=dataset.id
# )

# # Retrieving experiment data from the langfuse experiments API.

# experiment_items = langfuse.api.experiments.list_items(
#     from_start_time=dataset.created_at,
#     dataset_id=dataset.id,
#     experiment_id=experiment_id,
# )

# # Accessing traces observations through the langfuse observations API.

# for item in experiment_items.data:
#     observation = langfuse.api.observations.get_many(
#         fields="core,basic,time,metrics,metadata,model,usage",
#         trace_id=item.trace_id,
#         name="ChatOpenAI",
#         limit=100,
#     )
#     print(f"Trace ID: {item.trace_id}, Observations: {len(observation.data)}")

# # Access the scores for the experiment through the langfuse scores API.

# scores_experiment = langfuse.api.scores_v3.get_many_v3(
#     experiment_id=experiment_id, name="avg_accuracy"
# )
# print(
#     f"Scores for experiment {experiment_id}: {scores_experiment.data[0].value if scores_experiment.data else 'No scores found'}"
# )
