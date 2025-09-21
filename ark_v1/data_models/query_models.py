from pydantic import BaseModel
from typing import List
from ark_v1.data_models.graph_models import (
    NodeVerified,
    EdgeVerified,
    TripleVerified,
)


class QueryResult(BaseModel):
    """A model to represent the result of a query to the graph."""

    success: bool = True
    errors: str = ""


class QueryResultEntitiesExist(QueryResult):
    """A model to represent the result of checking if entities exist in the graph."""

    results: List[NodeVerified] = []


class QueryResultRelationsExist(QueryResult):
    """A model to represent the result of checking if relations exist in the graph."""

    results: List[EdgeVerified] = []


class QueryResultGetRelations(QueryResult):
    """A model to represent the result of retrieving relations from the graph."""

    results: List[str] = []


class QueryResultGetEdges(QueryResult):
    """A model to represent the result of retrieving edges from the graph."""

    results: List[EdgeVerified] = []


class QueryResultGetTriples(QueryResult):
    """A model to represent the result of retrieving triples from the graph."""

    results: List[TripleVerified] = []
