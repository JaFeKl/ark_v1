import sys
import os
import json
import logging
from ark_v1.ark_v1 import ARK_V1
from langchain_ollama import ChatOllama


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python minimal_ollama.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"Using model: {model_name}")

    llm = ChatOllama(
        model="qwen3:14b",
        base_url="http://localhost:11434",
        temperature=0.9,
        top_p=0.9,
        min_p=0.0,
        top_k=40,
        seed=42,
    )

    agent = ARK_V1()
    agent.llm = llm
    agent.load_configuration(config={})

    # load example data
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_data.json"),
        "r",
    ) as file:
        example_data = json.load(file)

    agent.load_graph_data(example_data["graph"])
    agent.set_initial_state(question=example_data["question"])
    final_state = agent.run()

    logging.info(
        f"Final answer: {final_state.get('finalAnswer').get('answer', 'No answer found')}"
    )


if __name__ == "__main__":
    main()
