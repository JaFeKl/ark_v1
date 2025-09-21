import os
import re
import getpass
import hashlib
import random
from typing import List, Any, Dict
from langgraph.graph.state import CompiledStateGraph
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


OPENAI_MODELS = [
    "gpt-5-mini",
    "gpt-5-nano-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5",
    "gpt-4o",
    "gpt-5-chat-latest",
]
OPEN_ROUTER_MODELS = [
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
    "google/gemini-2.5-flash",
    "openai/gpt-5-chat",
    "openai/gpt-oss-120b",
]
OLLAMA_MODELS = ["llama3.2:latest", "qwq:latest", "qwen3:14b", "qwen3:8b"]


def get_llm(
    name: str = "qwq:latest",
    temperature: float = 0.1,
    top_p: float = 0.95,
    min_p: float = 0.0,
    top_k: int = 40,
    seed: int = 42,
):
    if name in OPENAI_MODELS:
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass(
                "Enter your OpenAI API key: "
            )
        return ChatOpenAI(
            model=name,
            temperature=temperature,  # Set temperature
            # seed=seed,  # Set seed
            output_version="responses/v1",
            reasoning={"effort": "minimal"},
            max_completion_tokens=300,
        )
    elif name in OLLAMA_MODELS:
        return ChatOllama(
            model=name,
            base_url="http://localhost:11434",
            temperature=temperature,  # Set temperature
            top_p=top_p,  # Set TopP
            min_p=min_p,  # Set MinP
            top_k=top_k,  # Set TopK
            seed=seed,  # Set seed
        )  # No repetition penalty)
    elif name in OPEN_ROUTER_MODELS:
        if not os.environ.get("OPENROUTER_API_KEY"):
            os.environ["OPENROUTER_API_KEY"] = getpass.getpass(
                "Enter your OpenRouter API key: "
            )
        return ChatOpenAI(
            model=name,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=temperature,  # Set temperature
            model_kwargs={
                "top_p": top_p,  # Set TopP
            },
            seed=seed,
        )


def filter_think_tags(output: str) -> str:
    """
    Removes the content inside <think>...</think> tags from the output.

    Args:
        output (str): The reasoning model's output.

    Returns:
        str: The filtered output without <think>...</think> content.
    """
    # Use a regex to remove <think>...</think> and the tags themselves
    filtered_output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
    return filtered_output.strip()


def get_small_llm():
    return ChatOllama(
        model="llama3.2:latest",
        base_url="http://localhost:11434",
    )


def get_knowledge_graph_llm():
    return get_llm(name="sebdg/valuechainhackers:latest")


def set_api_key(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


def pretty_print(output):
    pass


def list_merge(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Merge two lists for concurrent state updates.

    Args:
        list1: First list
        list2: Second list

    Returns:
        List[Any]: Combined list
    """
    if not list1:
        return list2
    if not list2:
        return list1
    return list1 + list2


def dict_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    return {**dict1, **dict2}


def draw_graph(graph: CompiledStateGraph, path="test.png"):
    dot = graph.get_graph()
    dot.draw_png(path)


def generate_4_digit_hash(input_string: str) -> str:
    # Define a custom character set including numbers, letters, and symbols
    custom_charset = (
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#@$%&*"
    )

    # Generate a hash using SHA256
    hash_object = hashlib.sha256(input_string.encode())
    full_hash = hash_object.hexdigest()

    # Convert the hash to an integer and map it to the custom character set
    random.seed(int(full_hash, 16))  # Seed the random generator with the hash
    return "".join(random.choices(custom_charset, k=4))  # Generate a 4-character hash
