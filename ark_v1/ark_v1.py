import json
from typing import Optional, Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from ark_v1.utils import (
    draw_graph,
    get_llm,
    generate_4_digit_hash,
)

from ark_v1.agent import Agent

from ark_v1.data_models.query_models import (
    QueryResultEntitiesExist,
    QueryResultGetEdges,
    QueryResultRelationsExist,
    QueryResultGetTriples,
)

from ark_v1.graph.graph_interface import (
    GraphInterface,
)

import ark_v1.chat_prompt_templates as prompts
from ark_v1.data_models.agent_models import (
    AgentState,
    QuestionTypes,
    RuntimeState,
    ReasoningStep,
    Anchor,
    AnchorVerified,
    Relation,
    RelationVerified,
    ReasoningStepResult,
    ReasoningStepResultVerified,
    FinalAnswerYesNo,
)


class ARK_V1(Agent):
    """A multi-hop reasoning agent that uses a knowledge graph to answer user requests."""

    def __init__(
        self,
        question: str = None,
        question_type=QuestionTypes.YES_NO,
        draw_graph: bool = False,
    ):
        super().__init__(
            name="ark_v1",
        )
        self._graph_interface = GraphInterface()
        self._question_type = question_type
        self.graph: CompiledStateGraph = self._create_graph(draw=draw_graph)
        if question:
            self.set_initial_state(question)
        self.logger.info("Initialized knowledge graph agent.")

    def load_configuration(
        self, config: Dict[str, Any], random_seed: Optional[int] = None
    ) -> None:
        """Load the configuration from a dictionary, commonly retrieved from a database"""

        # Check if the config contains prompt templates
        self._load_llm(config, random_seed=random_seed)
        self._add_tools(self.llm)
        self._load_prompt_templates()
        self.max_reasoning_steps = config.get("max_reasoning_steps", 8)
        self.max_attempts = config.get("max_attempts", 5)
        self.recursion_limit = config.get("recursion_limit", 200)

    def load_configuration_from_file(self, file_path: str) -> None:
        """Load the configuration from a JSON file"""
        with open(file_path, "r") as file:
            config = json.load(file)
        self.load_configuration(config)

    def load_graph_data(self, graph_data: List[List]) -> None:
        if graph_data:
            self._graph_interface.load_graph_data(graph_data)
            self.logger.info(f"Set Graph with {len(graph_data)} entries.")

    def _load_llm(
        self, config: Dict[str, Any], random_seed: Optional[int] = None
    ) -> None:
        """Load the LLM from the configuration"""
        if "llm" not in config:
            if self.llm is None:
                self.logger.warning(
                    "Caution: No LLM configuration found in the provided config and no LLM is set."
                )
            else:
                return
        else:
            llm_config = config["llm"]
            if "model" not in llm_config:
                raise ValueError("LLM configuration must contain a 'model' field.")
            llm_kwargs = {"name": llm_config["model"]}
            if "temperature" in llm_config:
                llm_kwargs["temperature"] = llm_config["temperature"]
            if "top_p" in llm_config:
                llm_kwargs["top_p"] = llm_config["top_p"]
            if "min_p" in llm_config:
                llm_kwargs["min_p"] = llm_config["min_p"]
            if "top_k" in llm_config:
                llm_kwargs["top_k"] = llm_config["top_k"]
            if "seed" in llm_config:
                llm_kwargs["seed"] = llm_config["seed"]
            elif random_seed is not None:
                llm_kwargs["seed"] = random_seed

            llm_object = get_llm(**llm_kwargs)
            self.llm = llm_object

    def _load_prompt_templates(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            self._add_prompt_templates()
        else:
            for template_name, template_config in config["prompt_templates"].items():
                self.add_prompt_template(
                    template_name,
                    ChatPromptTemplate.from_messages(template_config["messages"]),
                )

    def _add_tools(self, llm: ChatOllama):
        for tool_name, tool in self._graph_interface.tool_registry.items():
            self.add_tool(tool_name, tool)
        llm.bind_tools(tools=self.tool_registry.values())

    def _add_prompt_templates(self):
        """static method to add required prompt templates from template file"""
        self.add_prompt_template(
            "system_description", prompts.get_prompt_system_description()
        )
        self.add_prompt_template("select_anchor", prompts.get_prompt_select_anchor())
        self.add_prompt_template(
            "select_relation", prompts.get_prompt_select_relation()
        )
        self.add_prompt_template(
            "perform_reasoning", prompts.get_prompt_perform_reasoning()
        )
        self.add_prompt_template(
            "summarize_reasoning", prompts.get_prompt_summarize_reasoning()
        )
        self.add_prompt_template(
            "create_final_answer", prompts.get_prompt_create_final_answer()
        )

    def set_initial_state(
        self, question: str, question_type: QuestionTypes = QuestionTypes.YES_NO
    ) -> None:
        """Set the initial state including the user's question and the system description for the agent"""
        self._question_type = question_type
        system_description = self.get_message_from_prompt_template("system_description")
        self.initial_state = RuntimeState(question=question)
        self.initial_state.extend_messages(system_description)
        self.initial_state.append_message(HumanMessage(content=question))

        # vectorize the subgraph
        self._graph_interface.graph.vectorize()

    def _create_graph(self, draw: bool = True) -> CompiledStateGraph:

        def initialize_reasoning_step(state: RuntimeState) -> RuntimeState:
            """Initialize a new reasoning step"""
            state.iteration += 1
            state.currentState = AgentState.NEW_REASONING_STEP
            state.reasoningSteps.append(ReasoningStep(anchor=AnchorVerified()))
            return state

        # Anchor Selection

        def prompt_select_anchor(state: RuntimeState) -> RuntimeState:
            current_reasoning_step = state.reasoningSteps[-1]
            current_anchor: AnchorVerified = current_reasoning_step.anchor

            failed_attempts_prompt = ""
            if current_anchor.attempt > 1:
                failed_attempts_prompt += f"You previously selected the anchor candidate {current_anchor.anchor.value} which does not exist in the knowledge graph. \n"
                for entity in current_anchor.verificationResults.results:
                    if not entity.verified:
                        failed_attempts_prompt += f"Closely related existing entities to candidate {entity.value}: {entity.alternatives}\n"

            prompt = self.get_message_from_prompt_template(
                "select_anchor",
                {
                    "iteration": state.iteration,
                    "attempt": current_anchor.attempt,
                    "failed_attempts": failed_attempts_prompt,
                    "schema": json.dumps(Anchor.model_json_schema()),
                },
            )
            state.extend_messages(prompt)
            return state

        def select_anchor(state: RuntimeState) -> RuntimeState:
            """Select an anchor node"""
            state.currentState = AgentState.SELECT_ANCHOR
            current_reasoning_step = state.reasoningSteps[-1]
            current_anchor: AnchorVerified = current_reasoning_step.anchor
            current_anchor.attempt += 1

            # use llm to generate an anchor candidate
            current_anchor.anchor = self.llm.with_structured_output(Anchor).invoke(
                state.messages
            )

            # verify that anchor candidate exists
            tool_result: QueryResultEntitiesExist = self.get_tool_by_name(
                "entities_exist"
            ).invoke(
                {
                    "entities": [current_anchor.anchor.value],
                    "retrieve_alternatives": True,
                }
            )

            current_anchor.verificationResults = tool_result

            # all entities in current_anchor.verificationResults.results must have the verified field set to True
            current_anchor.valid = all(
                entity.verified for entity in current_anchor.verificationResults.results
            )

            state.append_message(
                AIMessage(
                    content=f"Result (Iteration {state.iteration}, Step 2.1., Attempt {current_anchor.attempt}): \nValid: {current_anchor.valid} \nSelected anchor entity: {current_anchor.anchor.value}",
                )
            )

            return state

        def _route_select_anchor(state: RuntimeState) -> str:
            """Route based on the validity of the anchor candidate"""
            current_reasoning_step = state.reasoningSteps[-1]
            if current_reasoning_step.anchor.attempt >= self.max_attempts:
                return "generate_answer"
            elif current_reasoning_step.anchor.valid:
                return "retrieve_relations"
            else:
                return "prompt_select_anchor"

        # Relation Selection

        def retrieve_relations(state: RuntimeState) -> RuntimeState:
            current_reasoning_step = state.reasoningSteps[-1]
            anchor_value: str = current_reasoning_step.anchor.anchor.value

            edges: QueryResultGetEdges = self.get_tool_by_name("get_edges").invoke(
                {"entities": [anchor_value]}
            )
            if not edges.success:
                raise ValueError(
                    f"Failed to retrieve relations for anchor entity {anchor_value}: {edges.errors}"
                )
            current_reasoning_step.relations_retrieved = edges.results

            state.append_message(
                SystemMessage(
                    content=f"Relations connected to entity {anchor_value}: {list(set(edge.get_relation() for edge in edges.results))}",
                )
            )
            return state

        def _route_retrieve_relations(state: RuntimeState) -> str:
            """Route the state based on the retrieved relations"""
            current_reasoning_step = state.reasoningSteps[-1]
            if len(current_reasoning_step.relations_retrieved) > 0:
                return "prompt_select_relation"
            else:
                return "prompt_select_anchor"

        def prompt_select_relation(state: RuntimeState) -> RuntimeState:
            current_reasoning_step = state.reasoningSteps[-1]

            if not current_reasoning_step.relation_selected:
                # If no relations were selected yet, we initialize the relations_selected field
                current_reasoning_step.relation_selected = RelationVerified()

            relation_verified = current_reasoning_step.relation_selected

            summary = ""
            if relation_verified.attempt > 1:
                summary = f'You have previously attempted to select the relation "{relation_verified.candidate.value}". The semantically closest relations to your selection were: {relation_verified.edge.alternatives}. \n'

            # Add the prompt for selecting relation candidates
            prompt = self.get_message_from_prompt_template(
                "select_relation",
                {
                    "iteration": state.iteration,
                    "attempt": relation_verified.attempt,
                    "max_number_of_relations": 3,
                    "summary_previous_attempts": summary,
                    "schema": json.dumps(Relation.model_json_schema()),
                },
            )
            state.extend_messages(prompt)
            return state

        def select_relation(state: RuntimeState) -> RuntimeState:
            state.currentState = AgentState.SELECT_RELATION
            current_reasoning_step = state.reasoningSteps[-1]
            relation_verified = current_reasoning_step.relation_selected
            relation_verified.attempt += 1

            try:
                relations_selected: Relation = self.llm.with_structured_output(
                    Relation
                ).invoke(
                    state.messages
                )  # get last
                relation_verified.candidate = relations_selected

            except ValueError as e:
                # If the LLM fails to select relations, we set the relations to invalid
                state.append_message(
                    AIMessage(
                        content=f"Failed to select relations. You have selected a wrong key. Full error message {e}.",
                    )
                )
                relation_verified.valid = False
                return state

            # check if the relation exists in the graph
            tool_result: QueryResultRelationsExist = self.get_tool_by_name(
                "relations_exist"
            ).invoke(
                {
                    "relations": [relations_selected.value],
                    "head": current_reasoning_step.anchor.anchor.value,
                    "retrieve_alternatives": True,
                    "num_alternatives": 2,
                }
            )
            relation_verified.edge = tool_result.results[0]
            relation_verified.valid = relation_verified.edge.verified

            state.append_message(
                AIMessage(
                    content=f"Result (Iteration {state.iteration}, Step 2.2., Attempt {relation_verified.attempt}): \nValid: {relation_verified.valid} \nSelected Relation: {relation_verified.candidate.value}"
                )
            )
            return state

        def _route_select_relation(state: RuntimeState) -> str:
            """Route the state to the next step based on the relation candidates validation"""
            current_reasoning_step = state.reasoningSteps[-1]
            if current_reasoning_step.relation_selected.attempt >= self.max_attempts:
                return "generate_answer"
            elif current_reasoning_step.relation_selected.candidate is None:
                return "select_anchor"
            elif current_reasoning_step.relation_selected.candidate.value == "":
                return "select_anchor"
            elif current_reasoning_step.relation_selected.valid is True:
                # If the relation candidates are valid, we continue with retrieving knowledge
                return "retrieve_triples"
            else:
                return "prompt_select_relation"

        # Triple Selection

        def retrieve_triples(state: RuntimeState) -> RuntimeState:
            """An automated node that retrieves knowledge from the knowledge graph"""
            current_reasoning_step = state.reasoningSteps[-1]

            # We expect AI messages the last message to be an AI message with the selected anchor entity
            retrieved_triples: QueryResultGetTriples = self.get_tool_by_name(
                "get_triples"
            ).invoke(
                {
                    "anchorEntity": current_reasoning_step.anchor.anchor.value,
                    "relations": [
                        current_reasoning_step.relation_selected.candidate.value
                    ],
                }
            )

            if not retrieved_triples.success:
                raise ValueError(
                    f"Failed to retrieve triples for anchor entity {current_reasoning_step.anchor.anchor.value} and relations {current_reasoning_step.relation_selected.candidate.value}: {retrieved_triples.errors}"
                )

            # add hash to each triple
            for triple in retrieved_triples.results:
                triple.key = generate_4_digit_hash(
                    f"{triple.head.value}, {str(triple.edge)}, {triple.tail.value}"
                )

            # for each triple create a unique hash and store a dict representation
            current_reasoning_step.triples_retrieved = retrieved_triples.results

            state.append_message(
                SystemMessage(
                    content="Retrieved Triples: \n"
                    + str(current_reasoning_step.get_triples_retrieved()),
                )
            )
            state.currentState = AgentState.RETRIEVE_TRIPLES
            return state

        def reasoning(state: RuntimeState) -> RuntimeState:
            """The LLM is invoked to reason about the knowledge it retrieved"""
            current_reasoning_step = state.reasoningSteps[-1]
            if current_reasoning_step.result is None:
                current_reasoning_step.result = ReasoningStepResultVerified()

            triples_retrieved = current_reasoning_step.triples_retrieved
            verified_result = current_reasoning_step.result
            verified_result.attempt += 1

            prompt = self.get_message_from_prompt_template(
                "perform_reasoning",
                {
                    "iteration": state.iteration,
                    "attempt": verified_result.attempt,
                    "allowed_keys": current_reasoning_step.get_keys_triples_retrieved(),
                    "schema": json.dumps(ReasoningStepResult.model_json_schema()),
                },
            )
            state.extend_messages(prompt)

            try:
                verified_result.result = self.llm.with_structured_output(
                    ReasoningStepResult
                ).invoke(state.messages)
            except Exception as e:
                print(f"Failed to reason: {e}")
                verified_result.valid = False
                state.append_message(
                    AIMessage(
                        content=f"While trying to generate a reasoning step you used wrongly formatted keys. The only allowed keys in this step are {current_reasoning_step.get_keys_triples_retrieved()}. Choose either one of these keys or set an empty list as the keys to reset the reasoning step."
                    )
                )
                return state

            triples_retrieved_dict = current_reasoning_step.get_dict_triples_retrieved()
            for key in verified_result.result.keys:
                # copy matching relations from the retrieved relations into relation_results.relations
                triple = triples_retrieved_dict.get(key, None)
                if triple is not None:
                    verified_result.triples.append(triple)
                else:
                    verified_result.invalid_keys.append(key)

            invalid_keys_str = ""
            if len(verified_result.invalid_keys) > 0:
                # If there are invalid triples, we set the reasoning step to invalid
                verified_result.valid = False
                invalid_keys_str = f" \nInvalid keys found: {verified_result.invalid_keys}. Please adjust your reasoning step."
                state.append_message(
                    AIMessage(
                        content=f"While trying to generate a reasoning step you used invalid keys {verified_result.invalid_keys}. The only allowed keys in this step are {current_reasoning_step.get_keys_triples_retrieved()}. Choose either one of these keys or set an empty list as the keys to reset the reasoning step."
                    )
                )
            else:
                # If all triples are valid, we set the reasoning step to valid
                verified_result.valid = True
                state.append_message(
                    AIMessage(
                        content=f"Result (Iteration {state.iteration}, Step 4, Attempt {verified_result.attempt}): \nValid: {verified_result.valid} \nSelected Triples: {[triple.get_tuple() for triple in verified_result.triples]} \nImplications: {verified_result.result.implications}{invalid_keys_str} \nContinue Exploration: {verified_result.result.continue_exploration} \nReset Reasoning Step: {verified_result.result.reset_reasoning_step}"
                    )
                )
            current_reasoning_step.finished = True
            return state

        def _route_reasoning_step(state: RuntimeState) -> str:
            """Evaluate the reasoning step and decide what to do next"""
            current_reasoning_step = state.reasoningSteps[-1]

            # We first check if the selected triples for the reasoning step are invalid
            if (
                state.iteration >= self.max_reasoning_steps
                or current_reasoning_step.result.attempt >= self.max_attempts
            ):
                return "generate_answer"
            elif not current_reasoning_step.result.valid:
                return "retrieve_triples"
            else:
                return "cleanup"

        def cleanup(state: RuntimeState) -> RuntimeState:
            """Cleanup reasoning process"""
            # set global continue exploration flag
            current_reasoning_step = state.reasoningSteps[-1]
            if state.iteration >= self.max_reasoning_steps:
                state.continue_exploration = False
            elif current_reasoning_step.result.result.reset_reasoning_step is True:
                state.continue_exploration = False
            else:
                state.continue_exploration = (
                    current_reasoning_step.result.result.continue_exploration
                )

            if (
                current_reasoning_step.result.result.reset_reasoning_step is True
                or current_reasoning_step.result.triples == []
            ):
                state.reasoningSteps.pop()

            human_message_index = next(
                (
                    i
                    for i, message in enumerate(state.messages)
                    if isinstance(message, HumanMessage)
                ),
                None,
            )
            # remove all messages following the human message
            if human_message_index is not None:
                state.messages = state.messages[: human_message_index + 1]

            # create summary of the reasoning steps
            prompt = self.get_message_from_prompt_template(
                "summarize_reasoning",
                {
                    "number_of_steps": len(state.reasoningSteps),
                    "summary": _get_reasoning_step_summary(state),
                },
            )
            state.extend_messages(prompt)
            return state

        def _route_continue_exploration(state: RuntimeState) -> str:
            """Route the state based on the global continue exploration flag"""
            if state.continue_exploration is True:
                return "initialize_reasoning_step"
            else:
                return "generate_answer"

        def generate_answer(state: RuntimeState) -> RuntimeState:
            """Generate the final answer based on the reasoning steps and the question type"""
            answer = FinalAnswerYesNo
            question_type_str = "The user request is a yes/no question. Therefore please set the finalAnswer accordingly."
            if self._question_type == QuestionTypes.YES_NO:
                question_type_str = "The user request is a yes/no question. Therefore please set the finalAnswer accordingly."
                answer = FinalAnswerYesNo
            else:
                NotImplementedError(
                    f"Question type {self._question_type} is not yet implemented yet."
                )

            prompt = self.get_message_from_prompt_template(
                "create_final_answer",
                {
                    "user_request": state.question,
                    "schema": json.dumps(answer.model_json_schema()),
                    "reasoning_steps": _get_reasoning_step_summary(state),
                    "question_type_str": question_type_str,
                },
            )
            state.extend_messages(prompt)
            result = self.llm.with_structured_output(answer).invoke(state.messages)
            state.append_message(AIMessage(content=result.model_dump_json()))
            state.finalAnswer = result
            return state

        # Utility

        def _get_reasoning_step_summary(state: RuntimeState) -> str:
            """Summarize the reasoning steps so far"""
            reasoning_steps_str = "Detailed Reasoning Steps:\n"
            for i, step in enumerate(state.reasoningSteps):
                if step.finished is False:
                    continue
                reasoning_steps_str += f"Reasoning Step {i+1}: Used Triples: {step.result.get_triples_list()}, Implications: {step.result.result.implications} \n\n"

            return reasoning_steps_str

        graph_builder = StateGraph(RuntimeState)

        # anchor
        graph_builder.add_node("initialize_reasoning_step", initialize_reasoning_step)
        graph_builder.add_node("prompt_select_anchor", prompt_select_anchor)
        graph_builder.add_node("select_anchor", select_anchor)

        # relations
        graph_builder.add_node("retrieve_relations", retrieve_relations)
        graph_builder.add_node("prompt_select_relation", prompt_select_relation)
        graph_builder.add_node("select_relation", select_relation)

        # triples and reasoning
        graph_builder.add_node("retrieve_triples", retrieve_triples)
        graph_builder.add_node("reasoning", reasoning)

        # continuation and cleanup
        graph_builder.add_node("cleanup", cleanup)
        graph_builder.add_node("generate_answer", generate_answer)

        # Connections
        graph_builder.add_edge(START, "initialize_reasoning_step")
        graph_builder.add_edge("initialize_reasoning_step", "prompt_select_anchor")
        graph_builder.add_edge("prompt_select_anchor", "select_anchor")
        graph_builder.add_conditional_edges(
            "select_anchor",
            _route_select_anchor,
            [
                "prompt_select_anchor",
                "retrieve_relations",
                "generate_answer",
            ],
        )
        graph_builder.add_conditional_edges(
            "retrieve_relations",
            _route_retrieve_relations,
            [
                "prompt_select_relation",
                "prompt_select_anchor",
            ],
        )
        graph_builder.add_edge("prompt_select_relation", "select_relation")
        graph_builder.add_conditional_edges(
            "select_relation",
            _route_select_relation,
            [
                "select_anchor",
                "prompt_select_relation",
                "retrieve_triples",
                "generate_answer",
            ],
        )

        graph_builder.add_edge("retrieve_triples", "reasoning")
        graph_builder.add_conditional_edges(
            "reasoning",
            _route_reasoning_step,
            [
                "cleanup",
                "retrieve_triples",
                "generate_answer",
            ],
        )
        graph_builder.add_conditional_edges(
            "cleanup",
            _route_continue_exploration,
            [
                "initialize_reasoning_step",
                "generate_answer",
            ],
        )

        graph_builder.add_edge("generate_answer", END)
        graph = graph_builder.compile()

        if draw is True:
            draw_graph(graph, path="kg_agent_graph.png")
        return graph

    def run_with_config(self, config: Dict[str, Any]) -> RuntimeState:
        self.load_configuration(config)

    def run(self) -> dict:
        """Run the agent and return the final state"""
        events = self.graph.stream(
            self.initial_state,
            stream_mode="values",
            config={
                "recursion_limit": self.recursion_limit,
            },
        )

        last_event = None

        for event in events:
            if not event["messages"]:
                # Skip when messages are empty
                continue

            if last_event is not None and "messages_full" in event:
                if event["messages_full"] == last_event["messages_full"]:
                    # Skip when the messages are the same as the last event
                    continue

            last_message: AnyMessage = event["messages"][-1]
            # if isinstance(last_message, SystemMessage):
            #     # Skip system messages
            #     continue
            # elif isinstance(last_message, AIMessage) and last_message.content == "":
            #     # Skip empty AI messages
            #     continue

            print(
                f"\n--------  MesssageType: {last_message.__class__.__name__} --------"
            )
            print(f"{last_message.content}")

            last_event = event

            if event.get("finalAnswer") is not None:
                # print(f"Final answer: {event['finalAnswer']}")
                final_state = RuntimeState.model_validate(event)
                final_state_dump = final_state.model_dump()
                if isinstance(final_state.finalAnswer, FinalAnswerYesNo):
                    final_state_dump["finalAnswer"] = (
                        final_state.finalAnswer.model_dump()
                    )
                final_state_dump.pop("currentState")
                return final_state_dump
