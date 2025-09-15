"""
Base Agent Class for QuantAgent Framework
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseAgent(ABC):
    """
    Abstract base class for all QuantAgent agents.
    
    Each agent is responsible for a specific aspect of market analysis:
    - IndicatorAgent: Technical indicators calculation
    - PatternAgent: Chart pattern recognition
    - TrendAgent: Trend analysis
    - DecisionAgent: Final trading decision synthesis
    """
    
    def __init__(self, name: str):
        """
        Initialize the base agent.
        
        Args:
            name (str): Name of the agent
        """
        self.name = name
        self.last_analysis = None
        
    @abstractmethod
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Perform analysis on the provided market data.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            **kwargs: Additional parameters specific to each agent
            
        Returns:
            Dict[str, Any]: Analysis results in a structured format
        """
        pass
    
    def get_last_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Get the results of the last analysis performed.
        
        Returns:
            Optional[Dict[str, Any]]: Last analysis results or None
        """
        return self.last_analysis
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data has the required OHLCV columns.
        
        Args:
            data (pd.DataFrame): Market data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_columns)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return self.__str__()

