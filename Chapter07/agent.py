import json
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, Union
from uuid import uuid4

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

system_prompt = """You are an helpful flight assistant for Unity Airways, capable of supporting users with their bookings and policy questions using tools to retrieve information from Unity Airways knowledge bases.

Available user requests you can handle:
	•	Help customers find their flight reservations using booking reference (PNR), last name with email or phone, or ticket number.
	•	Assist in changing flight dates/times or canceling bookings, calculating and informing users about any applicable fees or waivers, and processing cancellations or changes.
	•	Answer customer questions about baggage allowances, change and refund rules, or disruption waivers by searching the policy FAQ.

You have access to these tools:
	•	lookup_booking: Given a PNR, ticket number, or name plus email/phone, retrieve full booking and policy context.
	•	modify_cancel_booking: Given booking context, change or cancel bookings, determine fees/waivers, and update booking state.
	•	faq_search: Given a natural language policy question, return the best answer, its source, and a confidence score.

When a customer asks a question:
	1.	Identify the primary intent: find booking, modify/cancel, or a policy query.
	2.	Gather necessary information (prompt the customer for any missing details).
	3.	Use the correct tool(s) to answer the request or perform the action.
	4.	For booking or change/cancel actions, clearly explain the outcome, including eligibility, rules, fees, refunds, or waivers.
	5.	If a policy query is made, search the FAQ and provide the most relevant, accurate information with attribution.
	6.	Escalate to a human agent if needed.
 
Always remain friendly, concise, and clear. Always confirm required information before taking any action. Make sure your responses are personalized and relevant to the user's intent. If the action asked cannot be handled, politely say so, and redirect to customer service contact email: support@unityairways.com. """

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## To create and see usage examples of more tools, see https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################
tools = []

# You can use UDFs in Unity Catalog as agent tools
UC_TOOL_NAMES = ["workspace.unity_airways.lookup_customer_info", "workspace.unity_airways.modify_cancel_booking"]
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
tools.extend(uc_toolkit.tools)

#############################
## Vector Search for FAQ Tool
#############################
# Use Databricks vector search indexes as tools
VECTOR_SEARCH_TOOLS = []

VECTOR_SEARCH_TOOLS.append(
    VectorSearchRetrieverTool(
        index_name="workspace.unity_airways.faq_index",
        num_results=5,
        tool_description="Search through unity airways frequently asked question (FAQ) about flight cancellation, travel policies and baggages security."
    )
)
tools.extend(VECTOR_SEARCH_TOOLS)

#####################
## Define agent logic
#####################

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]

def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
):
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent):
        self.agent = agent

    def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert from a Responses API output item to ChatCompletion messages."""
        msg_type = message.get("type")
        if msg_type == "function_call":
            return [
                {
                    "role": "assistant",
                    "content": "tool call",
                    "tool_calls": [
                        {
                            "id": message["call_id"],
                            "type": "function",
                            "function": {
                                "arguments": message["arguments"],
                                "name": message["name"],
                            },
                        }
                    ],
                }
            ]
        elif msg_type == "message" and isinstance(message["content"], list):
            return [
                {"role": message["role"], "content": content["text"]}
                for content in message["content"]
            ]
        elif msg_type == "reasoning":
            return [{"role": "assistant", "content": json.dumps(message["summary"])}]
        elif msg_type == "function_call_output":
            return [
                {
                    "role": "tool",
                    "content": message["output"],
                    "tool_call_id": message["call_id"],
                }
            ]
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        filtered = {k: v for k, v in message.items() if k in compatible_keys}
        return [filtered] if filtered else []

    def _prep_msgs_for_cc_llm(self, responses_input) -> list[dict[str, Any]]:
        "Convert from Responses input items to ChatCompletion dictionaries"
        cc_msgs = []
        for msg in responses_input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

    def _langchain_to_responses(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        "Convert from ChatCompletion dict to Responses output item dictionaries"
        for message in messages:
            message = message.model_dump()
            role = message["type"]
            if role == "ai":
                if tool_calls := message.get("tool_calls"):
                    return [
                        self.create_function_call_item(
                            id=message.get("id") or str(uuid4()),
                            call_id=tool_call["id"],
                            name=tool_call["name"],
                            arguments=json.dumps(tool_call["args"]),
                        )
                        for tool_call in tool_calls
                    ]
                else:
                    return [
                        self.create_text_output_item(
                            text=message["content"],
                            id=message.get("id") or str(uuid4()),
                        )
                    ]
            elif role == "tool":
                return [
                    self.create_function_call_output_item(
                        call_id=message["tool_call_id"],
                        output=message["content"],
                    )
                ]
            elif role == "user":
                return [message]

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = []
        for msg in request.input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

        for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
            if event[0] == "updates":
                for node_data in event[1].values():
                    for item in self._langchain_to_responses(node_data["messages"]):
                        yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
            # filter the streamed messages to just the generated text messages
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except Exception as e:
                    print(e)


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
mlflow.langchain.autolog()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphResponsesAgent(agent)
mlflow.models.set_model(AGENT)
