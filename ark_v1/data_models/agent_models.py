from enum import Enum
from typing import List, Optional, Annotated, Dict, Any
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import AnyMessage

from ark_v1.data_models.query_models import (
    QueryResult,
    QueryResultEntitiesExist,
    QueryResultGetRelations,
    QueryResultGetEdges,
    QueryResultRelationsExist,
    QueryResultGetTriples,
)
from ark_v1.data_models.graph_models import (
    Triple,
    EdgeVerified,
    TripleVerified,
)


class QuestionTypes(Enum):
    """Enum for existing question types"""

    YES_NO = "yes_no"
    OPEN_ENDED = "open_ended"
    MULTIPLE_CHOICE = "multiple_choice"
    ENTITY_LIST = "entity_list"
    OTHER = "other"


class AgentState(Enum):
    """Enum for different AgentState of the agent"""

    STARTING = "starting"
    SELECT_ANCHOR = "select_anchor"
    SELECT_RELATION = "select_relation"
    RETRIEVE_TRIPLES = "retrieve_triples"
    TOOLS = "tools"  # The state in which the agent has just called a tool
    FINAL = "final"
    NEW_REASONING_STEP = (
        "reasoning"  # This state signals that a new reasoing step should be initiated
    )


class VerifiedModel(BaseModel):
    """A base model for verified models that includes verified results"""

    verificationResults: Optional[QueryResult] = None
    valid: bool = False
    attempt: int = 0


class Keys(BaseModel):
    """A model that holds a list of keys"""

    keys: List[str] = Field(
        description="A list of keys that map to specific entities in a graph, e.g. triples. Keys always have exactly 4 characters.",
    )

    @field_validator("keys")
    def validate_key_length(cls, value):
        """Validate that each key has exactly 4 characters."""
        for item in value:
            if len(item) != 4:
                raise ValueError("Each key must have exactly 4 characters.")
        return value


# Anchor Models


class Anchor(BaseModel):
    """A node of the knowledge graph that has been selected to act as an anchor for exploration"""

    value: str = Field(
        description="The anchor node that should be used to explore the knowledge graph.",
        default="",
    )
    justification: str = Field(
        description="The reasoning behind selecting this anchor node.",
        default="",
    )


class AnchorVerified(VerifiedModel):
    """A verified anchor entity candidate that includes the verification results"""

    verificationResults: Optional[QueryResultEntitiesExist] = None
    anchor: Optional[Anchor] = Field(
        description="The anchor that was verified",
        default=None,
    )


# Relation Models


class Relation(BaseModel):

    value: str = Field(
        description="The relation that should be used to explore the knowledge graph.",
        default="",
    )
    justification: str = Field(
        description="The reasoning behind selecting this relation.",
        default="",
    )


class RelationVerified(VerifiedModel):
    """A model that holds the verified results of selected relations for a specific anchor entity"""

    candidate: Optional[Relation] = Field(
        description="A relation candidate selected by the LLM.",
        default=None,
    )
    edge: Optional[EdgeVerified] = Field(
        description="A verified edge of the knowledge graph that matches the relations.",
        default=None,
    )


# Triples models


class ReasoningStepResult(Keys):
    """A model that holds the keys for selected triples that were retrieved from the knowledge graph."""

    implications: str = Field(
        description="A short summary of implications that were drawn from the retrieved triples as well as future steps that should be taken to answer the user request. "
    )
    continue_exploration: bool = Field(
        description="Based on the implications, set this flag to indicate if the knowledge graph exploration should be continued."
    )
    reset_reasoning_step: bool = Field(
        description="If set to True, the reasoning step will be cancelled and a fresh reasoning step cycle is initiated.",
        default=False,
    )


class ReasoningStepResultVerified(VerifiedModel):
    """A model that holds the verified results of selected triples for a specific anchor entity and relations"""

    result: Optional[ReasoningStepResult] = Field(
        description="The inferred result of the reasoning step that includes the selected triples and implications.",
        default=None,
    )
    triples: List[TripleVerified] = Field(
        description="A list of verified triples that match the selected triples.",
        default=[],
    )
    invalid_keys: Optional[List[str]] = Field(
        description="A list of keys that were selected but are invalid.",
        default=[],
    )

    def get_triples_list(self) -> List[str]:
        """Get a string representation of the selected triples"""
        return [str(triple) for triple in self.triples]


class ReasoningStep(BaseModel):
    """A single reasoning step comprises the following components:
    1. A single anchor entity that is selected to explore the knowledge graph
    2. Relations that were retrieved from the knowledge graph based on the anchor entity
    3. A single relation that was selected from the retrieved relations to be used for exploration
    4. Triples that were retrieved from the knowledge graph based on the anchor entity and selected relation
    5. A reasoning result that includes the selected triples and implications
    """

    anchor: Optional[AnchorVerified] = None
    relations_retrieved: Optional[List[EdgeVerified]] = []
    relation_selected: Optional[RelationVerified] = None
    triples_retrieved: Optional[List[TripleVerified]] = None
    result: Optional[ReasoningStepResultVerified] = None
    finished: bool = False

    def get_triples_retrieved_str(self) -> str:
        """Get a string representation of the retrieved triples"""
        if self.triples_retrieved is None:
            return "No triples retrieved."
        return "\n".join([str(triple) for triple in self.triples_retrieved])

    def get_triples_retrieved(self) -> List[Dict[str, Any]]:
        """Get a list of dictionaries representing the retrieved triples"""
        if self.triples_retrieved is None:
            return []
        return [triple.get_dict() for triple in self.triples_retrieved]

    def get_keys_triples_retrieved(self) -> List[str]:
        """Get a list of keys for the retrieved triples"""
        if self.triples_retrieved is None:
            return []
        return [triple.key for triple in self.triples_retrieved]

    def get_relations_retrieved(self) -> List[str]:
        """Get a string representation of the retrieved relations"""
        if self.relations_retrieved is None:
            return []
        return [relation.get_relation() for relation in self.relations_retrieved]

    def get_keys_relations_retrieved(self) -> List[str]:
        """Get a list of keys for the retrieved relations"""
        if self.relations_retrieved is None:
            return []
        return [relation.key for relation in self.relations_retrieved]

    def get_dict_triples_retrieved(self) -> Dict[str, Triple]:
        """Get a dictionary representation of the retrieved triples"""
        if self.triples_retrieved is None:
            return {}
        return {triple.key: triple for triple in self.triples_retrieved}


# Answers


class FinalAnswer(BaseModel):
    """Final result of the reasoning process"""

    arisedQuestion: str = Field(
        description="A question that came up during the reasoning process that should be answered to improve your answer.",
        default="",
    )
    ambigoutyOfRequest: bool = Field(
        description="Flag indicating if the initial request is ambiguous and cant be answered directly",
    )
    justification: str = Field(
        description="The reasoning behind the final answer",
    )


class FinalAnswerYesNo(FinalAnswer):
    """Final answer for yes/no questions"""

    answer: Optional[bool] = Field(
        description="The final answer to the user request, either True or False. If the question couldn't be answered, it should be None.",
    )


class FinalAnswerEntityList(FinalAnswer):
    """Final answer for entity list questions"""

    answer: List[str] = Field(
        description="The final answer to the user request, a list of entities that answer the user request",
        default=[],
    )


class FinalAnswerEntityListVerified(VerifiedModel):
    """Final answer for entity list questions with verification results"""

    answer: FinalAnswerEntityList


# Runtime State


class RuntimeState(BaseModel):
    """The RuntimeState model that holds the answer to the given question"""

    # session_id: str
    question: str = Field(
        description="The user question that should be answered",
        default="",
    )
    currentState: AgentState = AgentState.STARTING
    messages: Annotated[
        List[AnyMessage], "The message history used in agent prompts"
    ] = []
    messages_full: Annotated[List[AnyMessage], "The full message history"] = []
    reasoningSteps: List[ReasoningStep] = Field(
        description="List of reasoning steps",
        default=[],
    )
    iteration: int = 0
    continue_exploration: bool = True
    finalAnswer: Optional[FinalAnswer] = None

    def append_message(self, message: AnyMessage) -> None:
        """Append a message to the message history"""
        self.messages.append(message)
        self.messages_full.append(message)

    def extend_messages(self, messages: List[AnyMessage]) -> None:
        """Extend the message history with a list of messages"""
        self.messages.extend(messages)
        self.messages_full.extend(messages)
