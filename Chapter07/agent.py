from typing import Annotated, Any, Generator, List, Optional, Sequence, TypedDict, Union

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient, DatabricksOAuthClientProvider
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolNode
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client as connect
from mlflow.pyfunc import ResponsesAgent
from pydantic import create_model

workspace_client = WorkspaceClient()
host = workspace_client.config.host
MANAGED_MCP_SERVER_URL = f"{host}/api/2.0/mcp/vector-search/workspace/unity_airways"

# Define a custom LangChain tool that wraps functionality for calling MCP servers
class MCPTool(BaseTool):
    """Custom LangChain tool that wraps MCP server functionality"""
    def __init__(
        self,
        name: str,
        description: str,
        args_schema: type,
        server_url: str,
        ws: WorkspaceClient,
    ):
        super().__init__(name=name, description=description, args_schema=args_schema)
        object.__setattr__(self, "server_url", server_url)
        object.__setattr__(self, "workspace_client", ws)

    def _run(self, **kwargs) -> str:
        """Execute the MCP tool"""
        # Use managed MCP server via synchronous call
        mcp_client = DatabricksMCPClient(
            server_url=self.server_url, workspace_client=self.workspace_client
        )
        response = mcp_client.call_tool(self.name, kwargs)
        return "".join([c.text for c in response.content])


# Convert an MCP tool definition into a LangChain-compatible tool
def create_langchain_tool_from_mcp(
    mcp_tool, server_url: str, ws: WorkspaceClient
):
    """Create a LangChain tool from an MCP tool definition"""
    schema = mcp_tool.inputSchema.copy()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Map JSON schema types to Python types for input validation
    TYPE_MAPPING = {"integer": int, "number": float, "boolean": bool}
    field_definitions = {}
    for field_name, field_info in properties.items():
        field_type_str = field_info.get("type", "string")
        field_type = TYPE_MAPPING.get(field_type_str, str)

        if field_name in required:
            field_definitions[field_name] = (field_type, ...)
        else:
            field_definitions[field_name] = (field_type, None)

    # Dynamically create a Pydantic schema for the tool's input arguments
    args_schema = create_model(f"{mcp_tool.name}Args", **field_definitions)
    print("args_schema: ", args_schema)
    # Return a configured MCPTool instance
    return MCPTool(
        name=mcp_tool.name,
        description=mcp_tool.description or f"Tool: {mcp_tool.name}",
        args_schema=args_schema,
        server_url=server_url,
        ws=ws
        )


mcp_tool = DatabricksMCPClient(server_url=MANAGED_MCP_SERVER_URL, workspace_client=workspace_client).list_tools()
tool = create_langchain_tool_from_mcp(mcp_tool[0], MANAGED_MCP_SERVER_URL, workspace_client)
