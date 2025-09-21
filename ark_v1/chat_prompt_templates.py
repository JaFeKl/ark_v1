import json
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate


## Prompts
def get_prompt_system_description() -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [
            (
                "system",
                "You are an AI agent that answers user questions by exploring a knowledge graph (KG) step-by-step. "
                "You are guided through the process of selecting and retrieving information to answer the request. "
                "Each step will be provided as a separate detailed system message. "
                "\n\nProcess:\n"
                "1. Receive the user request and (optionally) a summary of previous reasoning steps.\n"
                "2. Create a new reasoning step:\n"
                "   2.1 Select ONE specific anchor entity from the KG to explore next.\n"
                "   2.2 From the anchor entity's outgoing relations, select exactly ONE relation you want to follow.\n"
                "   2.3 The system retrieves triples for the selected relation. You reason strictly based on them.\n"
                "3. Decide whether to continue exploring by selecting a new anchor entity (if needed facts are missing) or produce a final answer.\n\n"
                "Rules:\n"
                "- Rely your reasoning on the retrieved information.\n"
                "- Always return results in the exact structured schema provided for that step.\n",
            )
        ]
    )


def get_prompt_summarize_reasoning() -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [
            (
                "system",
                "This is a summary of {number_of_steps} reasoning steps completed so far: "
                "{summary}\n\n"
                "You must use this summary to maintain consistency. Do not repeat facts already retrieved unless needed for clarity.",
            )
        ]
    )


def get_prompt_select_anchor() -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [
            (
                "system",
                "(Iteration {iteration}, Step 2.1, Attempt {attempt}): "
                "{failed_attempts}"
                "Task: Select exactly ONE specific anchor entity (a node name) from the knowledge graph "
                "that is most likely to lead to information for answering the user request.\n"
                "Rules:\n"
                "- Must be an EXACT match to a KG entity name.\n"
                "- Avoid special character changes (e.g., 'Maria de Ventadorn' must NOT become 'Maria_de_Ventadorn').\n"
                "Return your answer in the following structured schema:\n{schema}",
            )
        ]
    )


def get_prompt_select_relation() -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [
            (
                "system",
                "(Iteration {iteration}, Step 2.2, Attempt {attempt}): "
                "The previous message contains a list of relations starting from your selected anchor entity. "
                "{summary_previous_attempts}"
                "Task: Select a relation that you want to use to explore the graph towards answering the user question. "
                "Notes: "
                "- When returning an empty string as the relation value, you will be routed back to Step 2.1 to select a new anchor entity. Please state in justification why you want to do so.\n"
                "Return your answer in the following schema: \n{schema}",
            )
        ]
    )


def get_prompt_perform_reasoning() -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [
            (
                "system",
                "(Iteration {iteration}, Step 3, Attempt {attempt})\n"
                "You have received triples from the KG for your selected relations.\n"
                "Task: Reason based on these triples towards answering the user request.\n"
                "Rules:\n"
                "- You may ONLY reference triples retrieved in this step.\n"
                "- If previous reasoning steps exist, integrate them without contradiction.\n"
                "- Consider whether there are NEW potential anchor entities mentioned in the triples\n"
                "- Only set continueExploration = False if you are confident you can fully and unambiguously answer the user's request from the facts retrieved so far.\n"
                "- Select only keys from: {allowed_keys}.\n"
                "Return your answer in the following schema: \n{schema}",
            )
        ]
    )


def get_prompt_create_final_answer() -> ChatPromptTemplate:
    return ChatPromptTemplate(
        [
            (
                "system",
                "It is time to prepare the final answer for the user request: {user_request} "
                "The previous reasoning steps were as follows: \n"
                "{reasoning_steps}"
                "Please answer with a structured output given by the following schema: {schema}"
                "Notes on answering based on the question types:"
                " {question_type_str}",
            )
        ]
    )


def prompts_to_json(file_path: str):
    """
    Converts the prompts defined in this module to a JSON file.

    Args:
        file_path (str): The path where the JSON file will be saved.
    """

    def extract_messages(prompt: ChatPromptTemplate):
        # Convert the internal message tuples to dicts for JSON, ensuring serializability
        messages = []
        for message in prompt.messages:
            if isinstance(message, SystemMessagePromptTemplate):
                messages.append({"role": "system", "content": message.prompt.template})
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        return messages

    prompts = {
        "system_description": {
            "messages": extract_messages(get_prompt_system_description())
        },
        "select_anchor": {"messages": extract_messages(get_prompt_select_anchor())},
        "select_relation": {"messages": extract_messages(get_prompt_select_relation())},
        "perform_reasoning": {
            "messages": extract_messages(get_prompt_perform_reasoning())
        },
        "summarize_reasoning": {
            "messages": extract_messages(get_prompt_summarize_reasoning())
        },
        "create_final_answer": {
            "messages": extract_messages(get_prompt_create_final_answer())
        },
    }
    with open(file_path, "w") as f:
        json.dump(prompts, f, indent=4)
