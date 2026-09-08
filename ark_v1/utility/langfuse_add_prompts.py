from dotenv import load_dotenv
from langfuse import get_client
from ark_v1.chat_prompt_templates import (
    get_prompt_system_description,
    get_prompt_select_anchor,
    get_prompt_select_relation,
    get_prompt_perform_reasoning,
    get_prompt_summarize_reasoning,
    get_prompt_create_final_answer,
)
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    ChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv()
langfuse = get_client()

prompt_template = get_prompt_system_description()
print(prompt_template.messages)


def to_langfuse_chat(prompt_template: ChatPromptTemplate) -> list[dict[str, str]]:
    roles = {
        SystemMessagePromptTemplate: "system",
        HumanMessagePromptTemplate: "user",
        AIMessagePromptTemplate: "assistant",
    }

    messages = []
    for message in prompt_template.messages:
        if isinstance(message, ChatMessagePromptTemplate):
            role = message.role
        else:
            role = roles.get(type(message))

        if role is None:
            raise TypeError(f"Unsupported prompt message: {type(message)!r}")

        # Your current templates use simple {variable} placeholders.
        content = message.prompt.template.replace("{", "{{").replace("}", "}}")
        messages.append({"role": role, "content": content})

    return messages


langfuse.create_prompt(
    name="ark_v1_system_description",
    type="chat",
    prompt=to_langfuse_chat(get_prompt_system_description()),
    labels=["ark_v1", "system"],
    commit_message="Initial converted prompt from ChatPromptTemplate",
)

langfuse.create_prompt(
    name="ark_v1_select_anchor",
    type="chat",
    prompt=to_langfuse_chat(get_prompt_select_anchor()),
    labels=["ark_v1"],
    commit_message="Initial converted prompt from ChatPromptTemplate",
)

langfuse.create_prompt(
    name="ark_v1_select_relation",
    type="chat",
    prompt=to_langfuse_chat(get_prompt_select_relation()),
    labels=["ark_v1"],
    commit_message="Initial converted prompt from ChatPromptTemplate",
)

langfuse.create_prompt(
    name="ark_v1_perform_reasoning",
    type="chat",
    prompt=to_langfuse_chat(get_prompt_perform_reasoning()),
    labels=["ark_v1"],
    commit_message="Initial converted prompt from ChatPromptTemplate",
)

langfuse.create_prompt(
    name="ark_v1_summarize_reasoning",
    type="chat",
    prompt=to_langfuse_chat(get_prompt_summarize_reasoning()),
    labels=["ark_v1"],
    commit_message="Initial converted prompt from ChatPromptTemplate",
)

langfuse.create_prompt(
    name="ark_v1_create_final_answer",
    type="chat",
    prompt=to_langfuse_chat(get_prompt_create_final_answer()),
    labels=["ark_v1"],
    commit_message="Initial converted prompt from ChatPromptTemplate",
)
