from typing import List, Dict, Optional
from langchain_core.tools import tool, BaseTool
from ark_v1.graph.local_graph import (
    LocalGraph,
)
from ark_v1.data_models.graph_models import (
    NodeVerified,
    EdgeVerified,
    TripleVerified,
)

from ark_v1.data_models.query_models import (
    QueryResult,
    QueryResultEntitiesExist,
    QueryResultGetRelations,
    QueryResultGetEdges,
    QueryResultRelationsExist,
    QueryResultGetTriples,
)

GRAPH = LocalGraph()


@tool
def get_triples(
    anchorEntity: str,
    relations: Optional[List[str]] = None,
) -> QueryResultGetTriples:
    """Get a list of triples which relate the given entity which acts as a subject to other entiies"""
    try:
        triples = GRAPH.get_triples(anchorEntity, relations)
        results = []
        for triple in triples:
            head, edge_dict, tail = triple
            verified_triple = TripleVerified(
                head=NodeVerified(value=head, verified=True),
                edge=EdgeVerified(value=edge_dict, verified=True),
                tail=NodeVerified(value=tail, verified=True),
            )
            results.append(verified_triple.model_dump())

        return QueryResultGetTriples(
            results=results,
            success=True,
            errors="",
        )
    except Exception as e:
        return QueryResultGetTriples(
            results=[],
            success=False,
            errors=str(e),
        )


@tool
def triples_exist(
    triples: List[List[str]],
) -> QueryResult:
    """Check if the triples exist in the graph."""
    try:
        results = []
        for triple in triples:
            results.append(
                GRAPH.triple_exist(
                    head=triple[0],
                    relation=triple[1],
                    tail=triple[2],
                )
            )
        return QueryResult(
            results=results,
            success=True,
            errors="",
        )
    except Exception as e:
        return QueryResult(
            results=[],
            success=False,
            errors=str(e),
        )


@tool
def entities_exist(
    entities: List[str],
    retrieve_alternatives: bool = False,
    num_alternatives: int = 5,
) -> QueryResultEntitiesExist:
    """Check if the entities exist in the graph."""
    query_result = QueryResultEntitiesExist()
    try:
        results = GRAPH.entities_exist(entities)
        for i, entity in enumerate(entities):
            verified_entity = NodeVerified(
                value=entity,
                verified=results[i],  # No alternatives provided in this context
            )
            if verified_entity.verified is True:
                verified_entity.alternatives = None
            else:
                if retrieve_alternatives:
                    verified_entity.alternatives = GRAPH.get_close_nodes(
                        entity, num_alternatives
                    )
            query_result.results.append(verified_entity)
    except Exception as e:
        return QueryResultEntitiesExist(
            success=False,
            errors=str(e),
        )
    return query_result


@tool
def get_relations(
    entities: List[str],
) -> QueryResultGetRelations:
    """Get outgoing relations for a list of entities."""
    try:
        results = []
        for entity in entities:
            relations = GRAPH.get_relations(entity)
            results.extend(list(set(relations)))  # Remove duplicates
        return QueryResultGetRelations(
            results=results,
            success=True,
            errors="",
        )
    except Exception as e:
        return QueryResultGetRelations(
            results=[],
            success=False,
            errors=str(e),
        )


@tool
def get_edges(
    entities: List[str],
) -> QueryResultGetEdges:
    """Get outgoing edges for a list of entities."""
    try:
        results = []
        for entity in entities:
            edges = GRAPH.get_edges(entity)
            for edge in edges:
                verified_edge = EdgeVerified(value=edge, verified=True)
                results.append(verified_edge.model_dump())
        return QueryResultGetEdges(
            results=results,
            success=True,
            errors="",
        )
    except Exception as e:
        return QueryResultGetEdges(
            results=[],
            success=False,
            errors=str(e),
        )


@tool
def relations_exist(
    relations: List[str],
    head: Optional[str] = None,
    retrieve_alternatives: bool = False,
    num_alternatives: int = 5,
) -> QueryResultRelationsExist:
    """Check if the relations exist in the graph."""
    query_result = QueryResultRelationsExist()
    try:
        results = GRAPH.relations_exist(relations, head=head)
        for i, relation in enumerate(relations):
            verified_edge = EdgeVerified(
                value={"relation": relation},
                verified=results[i],  # No alternatives provided in this context
            )
            if verified_edge.verified is True:
                verified_edge.alternatives = None
            else:
                if retrieve_alternatives:
                    verified_edge.alternatives = GRAPH.get_close_relations(
                        relation, num_alternatives
                    )
            query_result.results.append(verified_edge)
    except Exception as e:
        return QueryResultRelationsExist(
            success=False,
            errors=str(e),
        )
    return query_result


class GraphInterface:
    def __init__(self):
        self.graph = GRAPH
        self.tool_registry: Dict[str, BaseTool] = {
            "get_triples": get_triples,
            "triples_exist": triples_exist,
            "entities_exist": entities_exist,
            "get_relations": get_relations,
            "get_edges": get_edges,
            "relations_exist": relations_exist,
        }

    def load_graph_data(self, graph_data: List):
        """Load graph data from a list of triples.

        The graph data is either formatted like this: [[head: str, relation: str, tail: str]]
        or like this [[head: str, tail:str, {"relation": str, "others": str}]]
        """
        self.graph.clear()
        for entry in graph_data:
            if len(entry) != 3:
                raise ValueError("Each entry must contain exactly three elements")

            if isinstance(entry[2], dict):
                head, tail, edge_dict = entry
                if "relation" not in edge_dict:
                    raise ValueError("The third element must contain a 'relation' key")
                self.graph.add_triple(head, tail, **edge_dict)

            elif isinstance(entry[2], str):
                head, tail, relation = entry
                edge_dict = {"relation": relation}
                self.graph.add_triple(head, tail, **edge_dict)

            else:
                raise ValueError(
                    "The third element must be either a string or a dictionary"
                )
