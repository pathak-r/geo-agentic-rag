"""
Agent Orchestration
LangChain agent that combines document search with production data tools.
"""
import pandas as pd
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import Optional

from src.llm import get_chat_llm, get_embeddings
from src.tools import (
    production_query_tool,
    anomaly_check_tool,
    calculate_recovery_factor,
    calculate_decline_rate,
    document_search_tool,
)

SYSTEM_PROMPT = """You are a Subsurface AI Assistant for the Volve oil field in the North Sea.
You help reservoir and production engineers analyze well data by combining:
1. Structured production data (oil rates, water cut, pressure, gas production)
2. Unstructured well documents (drilling reports, completion reports, final well reports)

You have access to tools for:
- Querying production data for any of the 7 Volve wells
- Detecting anomalies in production trends
- Calculating recovery factors and decline rates
- Searching well documentation (drilling reports, completion reports)

When answering questions:
- You are in a multi-turn chat: short follow-ups like "proceed", "use available data only", "same well", or "elaborate" refer to the previous user question—carry forward the well name, metric, or task from context and call tools as needed.
- Always specify which well(s) you're analyzing
- Cite specific data points and document sources
- If a question requires both production data AND document context, use both tools
- Flag any anomalies or concerning trends proactively
- Use subsurface engineering terminology appropriately
- When uncertain, say so — never fabricate data

The Volve field wells are: 15/9-F-1 C, 15/9-F-11 H, 15/9-F-12 H, 15/9-F-14 H,
15/9-F-15 D, 15/9-F-4 AH, 15/9-F-5 AH

The field produced from 2008 to 2016, with total production of ~63 million barrels.
The reservoir is in the Hugin Formation at approximately 2750-3120m depth.
"""


# --- Pydantic models for tool inputs ---

class ProductionQueryInput(BaseModel):
    well_name: str = Field(description="Well name or partial match, e.g. 'F-1' or 'F-11'")
    metric: Optional[str] = Field(default=None, description="Specific metric: BORE_OIL_VOL, BORE_WAT_VOL, WATER_CUT_PCT, AVG_WHP_P, GOR")
    start_date: Optional[str] = Field(default=None, description="Start date filter YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="End date filter YYYY-MM-DD")


class AnomalyCheckInput(BaseModel):
    well_name: Optional[str] = Field(default=None, description="Well name to check, or leave empty for all wells")


class RecoveryFactorInput(BaseModel):
    well_name: str = Field(description="Well name to calculate recovery for")
    ooip_sm3: Optional[float] = Field(default=None, description="Original Oil in Place in Sm3")


class DeclineRateInput(BaseModel):
    well_name: str = Field(description="Well name for decline analysis")
    period_months: int = Field(default=6, description="Number of months to analyze")


class DocumentSearchInput(BaseModel):
    query: str = Field(description="Natural language query to search well documents")


def create_agent(df: pd.DataFrame) -> AgentExecutor:
    """Create the LangChain agent with all tools."""
    llm = get_chat_llm(temperature=0.0)
    embeddings = get_embeddings()

    # Define tools
    tools = [
        StructuredTool.from_function(
            func=lambda well_name, metric=None, start_date=None, end_date=None:
                production_query_tool(df, well_name, metric, start_date, end_date),
            name="query_production_data",
            description="Query well production data. Use this for questions about oil/water/gas rates, "
                        "water cut, pressure, or production trends for specific wells.",
            args_schema=ProductionQueryInput,
        ),
        StructuredTool.from_function(
            func=lambda well_name=None: anomaly_check_tool(df, well_name),
            name="check_anomalies",
            description="Detect anomalies in production data. Use this to find unusual patterns "
                        "in water cut, production drops, pressure changes, or GOR spikes.",
            args_schema=AnomalyCheckInput,
        ),
        StructuredTool.from_function(
            func=lambda well_name, ooip_sm3=None:
                calculate_recovery_factor(df, well_name, ooip_sm3),
            name="calculate_recovery_factor",
            description="Calculate cumulative production and recovery factor for a well.",
            args_schema=RecoveryFactorInput,
        ),
        StructuredTool.from_function(
            func=lambda well_name, period_months=6:
                calculate_decline_rate(df, well_name, period_months),
            name="calculate_decline_rate",
            description="Calculate production decline rate for a well over a specified period.",
            args_schema=DeclineRateInput,
        ),
        StructuredTool.from_function(
            func=lambda query: document_search_tool(query, embeddings),
            name="search_well_documents",
            description="Search drilling reports, completion reports, and final well reports. "
                        "Use this for questions about drilling operations, well construction, "
                        "geological findings, formation data, or historical well events.",
            args_schema=DocumentSearchInput,
        ),
    ]

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
