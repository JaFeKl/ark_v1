from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class EntityVerified(BaseModel):
    """A model to represent a verified entity in the knowledge graph."""

    verified: bool = Field(
        description="Indicates if the entity is verified and exists in the graph",
        default=False,
    )
    alternatives: Optional[List[Dict[str, Any]]] = Field(
        description="Alternative entities and their similarity score based on embeddings",
        default=None,
    )


class Node(BaseModel):
    value: str = Field(
        description="The name of the node, e.g., 'Horsens'",
        default="",
    )

    def __str__(self) -> str:
        """String representation of the node entity"""
        return self.value


class NodeVerified(EntityVerified):
    """A model to represent a verified node in the knowledge graph."""

    value: str = Field(
        description="The name of the node, e.g., 'Horsens'",
        default="",
    )

    def __str__(self) -> str:
        """String representation of the node entity"""
        return self.value


class Edge(BaseModel):
    value: Dict = Field(
        description="The relation of the triple, e.g., {'relation': 'is located in'}",
        default={},
    )

    def get_relation(self) -> str:
        return self.value.get("relation", "")


class EdgeVerified(EntityVerified):
    """A model to represent a verified edge in the knowledge graph."""

    value: Dict = Field(
        description="The edge of the triple, e.g., {'relation': 'is located in'}",
        default={},
    )

    def get_relation(self) -> str:
        return self.value.get("relation", "")

    def __str__(self) -> str:
        return str(self.value)


class Triple(BaseModel):
    """A triple in the knowledge graph"""

    head: Node = Field(
        description="The head of the triple, e.g., 'Horsens'",
    )
    edge: Edge = Field(
        description="The edge of the triple, e.g., {'relation': 'is located in'}",
    )
    tail: Node = Field(
        description="The tail of the triple, e.g., 'city'",
    )
    key: Optional[str] = Field(
        description="A unique key value for the triple, used as a pointer to the respective triple.",
        default=None,
    )

    def get_dict(self) -> Dict[str, Any]:
        """Get a dictionary representation of the triple"""
        return {self.key: (self.head, self.edge, self.tail)}

    def get_triple(self):
        return (self.head, self.edge["relation"], self.tail)

    def __str__(self) -> str:
        """String representation of the triple"""
        return f"{self.get_dict()}"


class TripleVerified(BaseModel):
    """A model to represent a verified triple in the knowledge graph."""

    head: NodeVerified = Field(
        description="The head of the triple, e.g., 'Horsens'",
    )
    edge: EdgeVerified = Field(
        description="The edge of the triple, e.g., {'relation': 'is located in'}",
    )
    tail: NodeVerified = Field(
        description="The tail of the triple, e.g., 'city'",
    )
    key: Optional[str] = Field(
        description="A unique key value for the triple, used as a pointer to the respective triple.",
        default=None,
    )

    def __str__(self) -> str:
        """String representation of the verified triple"""
        return f"{self.head.value}, {self.edge.get_relation()}, {self.tail.value}"

    def get_tuple(self) -> tuple:
        """Get a tuple representation of the verified triple"""
        return (self.head.value, self.tail.value, self.edge.value)

    def get_dict(self) -> Dict[str, Any]:
        """Get a dictionary representation of the verified triple"""
        return {self.key: self.get_tuple()}
