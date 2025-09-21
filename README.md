# ARK-V1

ARK-V1 is our first version of a KG-Agent which explores a KG to answer natural language questions.

## Install

Install the agent using pip. E.g in a new conda environment:

```bash
conda create -n myenv python=3.11
conda activate myenv
pip install -e .
```

## Run the example

Test the agent by running a small example using a local qwen3:14b model:
#### Using a local Ollama model
```bash
python examples/minimal_ollama.py qwen3:14b
```