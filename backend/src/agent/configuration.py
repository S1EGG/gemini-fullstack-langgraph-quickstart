import os
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableField


class Configuration(BaseModel):
    """The configuration for the agent."""

    configurable_llm: ConfigurableField[str] = Field(
        title="Model",
        description="Model for Query Generator",
        default="gpt-4",
        enum=[
            "gpt-4",
            "gpt-4-turbo", 
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4o-mini"
        ],
    )

    reflection_llm: ConfigurableField[str] = Field(
        description="Model for Query Critic",
        default="gpt-4-turbo",
        enum=[
            "gpt-4",
            "gpt-4-turbo", 
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4o-mini"
        ],
    )

    answer_llm: ConfigurableField[str] = Field(
        description="Model for Query Answerer",
        default="gpt-4",
        enum=[
            "gpt-4",
            "gpt-4-turbo", 
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4o-mini"
        ],
    )

    max_search_results: ConfigurableField[int] = Field(
        description="Max number of search results per search",
        default=5,
        ge=1,
        le=10,
    )
    
    max_search_per_query: ConfigurableField[int] = Field(
        description="Max number of searches per query",
        default=2,
        ge=1,
        le=5,
    )
    
    number_of_initial_queries: ConfigurableField[int] = Field(
        description="Number of initial search queries to generate",
        default=3,
        ge=1,
        le=5,
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
