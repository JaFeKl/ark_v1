import logging
from typing import Dict, Optional, List, Union, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


class Agent:
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """The base class for all agents."""
        self.name = name
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.name}")

        self.llm: Optional[BaseChatModel] = None
        self.tool_registry: Dict[str, BaseTool] = {}
        self.prompt_templates: Dict[str, ChatPromptTemplate] = {}

    @property
    def llm(self) -> Optional[BaseChatModel]:
        """Get the LLM instance."""
        return self._llm

    @llm.setter
    def llm(self, value: BaseChatModel):
        if value is not None:
            if isinstance(value, ChatOpenAI):
                self.logger.info(f"Setting LLM: {type(value).__name__}, {value.name}")
            elif isinstance(value, ChatOllama):
                self.logger.info(f"Setting LLM: {type(value).__name__}, {value.model}")
            else:
                self.logger.warning(f"Setting LLM: {type(value).__name__}")
        self._llm = value

    def add_tool(self, key: str, tool: BaseTool):
        """Add a tool to the agent.

        Args:
            tool (BaseTool): The tool instance to be added.
        """
        self.tool_registry[key] = tool

    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Get a tool by its name.

        Args:
            name (str): The name of the tool to retrieve.

        Returns:
            Optional[BaseTool]: The tool function if found, otherwise None.
        """
        for tool in self.tool_registry.values():
            if tool.name == name:
                return tool
        return None

    def add_prompt_template(self, key: str, template: Union[ChatPromptTemplate, dict]):
        """Add a prompt template to the agent.
        Args:
            key (str): Custom key associated with the prompt template.
            template (Union[ChatPromptTemplate, dict]): The prompt template instance or a dictionary representation of it.
        """
        if isinstance(template, dict):
            template = self.load_chat_prompt_template(template)
        elif not isinstance(template, ChatPromptTemplate):
            raise ValueError(
                "Template must be a ChatPromptTemplate instance or a dictionary."
            )
        self.prompt_templates[key] = template

    def get_message_from_prompt_template(
        self, key: str, input: Dict = {}
    ) -> Optional[List[BaseMessage]]:
        output = self.prompt_templates.get(key).invoke(input)
        messages = output.to_messages()
        return messages

    def get_metadata(self):
        """Get metadata of all LLMs.

        Returns:
            dict: A dictionary containing metadata of all LLMs.
        """
        metadata = {}

        if isinstance(self.llm, ChatOllama):
            metadata["llm"] = self._get_metadata_llm_ollama(self.llm)

        for key, template in self.prompt_templates.items():
            prompts = {}
            if isinstance(template, ChatPromptTemplate):
                prompts[key] = {
                    "messages": [message.model_dump() for message in template.messages],
                    "input_variables": template.input_variables,
                    "output_parser": template.output_parser,
                }
            metadata["prompts"] = prompts
        return metadata

    def get_prompt_template(
        self, key: str, as_dict: bool = True
    ) -> Optional[Union[ChatPromptTemplate, dict]]:
        """Get a prompt template by its key.

        Args:
            key (str): The key associated with the prompt template.
            as_dict (bool): If True, return the template as a dictionary, otherwise return the ChatPromptTemplate object.

        Returns:
            Optional[Union[ChatPromptTemplate, dict]]: The prompt template if found, otherwise None.
        """
        template = self.prompt_templates.get(key, None)
        if template is not None:
            if as_dict:
                return self.chat_prompt_template_to_dict(template)
            else:
                return template
        return None

    def get_all_prompt_templates(
        self, as_dict: bool = True
    ) -> Dict[str, Union[ChatPromptTemplate, dict]]:
        """Get all prompt templates."""
        templates = []

        for key, template in self.prompt_templates.items():
            template = {}
            if as_dict:
                template[key] = templates.append(
                    self.chat_prompt_template_to_dict(template)
                )
            else:
                template[key] = template
        return templates

    def chat_prompt_template_to_dict(self, template: ChatPromptTemplate) -> dict:
        """Convert a ChatPromptTemplate to a dictionary.

        Args:
            chat_prompt_template (ChatPromptTemplate): The chat prompt template to convert.

        Returns:
            dict: A dictionary representation of the chat prompt template.
        """
        return template.model_dump()

    def load_chat_prompt_template(self, data: dict) -> ChatPromptTemplate:
        """Load a chat prompt template from a dictionary.

        Args:
            data (dict): A dictionary containing the prompt template data.

        Returns:
            ChatPromptTemplate: The loaded chat prompt template.
        """

        messages = []
        for msg in data["messages"]:
            msg_role = msg["role"]
            msg_content = msg["content"]
            messages.append((msg_role, msg_content))

        return ChatPromptTemplate(
            input_variables=data.get("input_variables"), messages=messages
        )

    def _get_metadata_llm_ollama(self, llm: ChatOllama):
        """Get metadata of the LLM from Ollama.

        Returns:
            dict: A dictionary containing metadata of the LLM from Ollama.
        """
        return (
            {
                "model_name": llm.model,  # Name of the LLM (e.g., ChatOllama)
                "temperature": llm.temperature,  # Temperature setting
                "seed": llm.seed,
                "top_p": llm.top_p,  # TopP setting
                "top_k": llm.top_k,  # TopK setting
                "tfs_z": llm.tfs_z,  # TFSZ setting
                "mirostat_tau": llm.mirostat_tau,  # Microstat tau setting
            },
        )
