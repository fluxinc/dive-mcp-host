"""Module for tracking MCP server interactions."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPServerTracker:
    """Class to track MCP server interactions including tool calls, results, and sources."""
    
    _instance = None
    
    def __new__(cls):
        """Ensure the MCPServerTracker is a singleton."""
        if cls._instance is None:
            cls._instance = super(MCPServerTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the MCPServerTracker."""
        if self._initialized:
            return
        
        self._chat_tool_calls = {}  # Maps chat_id to list of tool calls
        self._chat_tool_results = {}  # Maps chat_id to list of tool results
        self._chat_sources = {}  # Maps chat_id to list of sources
        self._initialized = True
    
    @classmethod
    def getInstance(cls):
        """Get the singleton instance of MCPServerTracker."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def track_tool_call(self, chat_id: str, tool_call: Dict[str, Any]):
        """Track a tool call for a specific chat.
        
        Args:
            chat_id: Unique identifier for the chat
            tool_call: The tool call details to track
        """
        if chat_id not in self._chat_tool_calls:
            self._chat_tool_calls[chat_id] = []
        self._chat_tool_calls[chat_id].append(tool_call)
        logger.debug(f"[{chat_id}] Tracked tool call: {tool_call['name']}")
    
    def track_tool_result(self, chat_id: str, tool_result: Dict[str, Any]):
        """Track a tool result for a specific chat and extract sources if present.
        
        Args:
            chat_id: Unique identifier for the chat
            tool_result: The tool result details to track
        """
        if chat_id not in self._chat_tool_results:
            self._chat_tool_results[chat_id] = []
        self._chat_tool_results[chat_id].append(tool_result)
        
        # If this is a query tool with sources, extract and track them
        if (tool_result.get('name') == 'query' and 
            tool_result.get('result') and 
            isinstance(tool_result['result'].get('content'), list)):
            
            # Find source content
            sources_item = None
            for item in tool_result['result']['content']:
                if (isinstance(item, dict) and 
                    item.get('type') == 'text' and 
                    item.get('text') and 
                    isinstance(item.get('text'), str) and
                    item['text'].startswith("<SOURCES>")):
                    sources_item = item
                    break
            
            if sources_item:
                logger.debug(f"[{chat_id}] Found sources in query result")
                # Source URLs format: <SOURCES><FILENAME>filename1</FILENAME>url1\n<FILENAME>filename2</FILENAME>url2\n...</SOURCES>
                source_text = sources_item['text'].replace("<SOURCES>", "").replace("</SOURCES>", "").strip()
                sources_list = source_text.split("\n")
                
                parsed_sources = []
                for item in sources_list:
                    if "</FILENAME>" in item:
                        split_source = item.split("</FILENAME>")
                        filename = split_source[0][len("<FILENAME>"):]
                        url = split_source[1]
                        parsed_sources.append({"filename": filename, "url": url})
                    else:
                        # Fallback for sources without filenames
                        parsed_sources.append({"filename": "", "url": item})
                
                self.track_sources(chat_id, parsed_sources)
    
    def track_sources(self, chat_id: str, sources: List[Dict[str, str]]):
        """Track sources for a specific chat, ensuring uniqueness.
        
        Args:
            chat_id: Unique identifier for the chat
            sources: List of source dictionaries with filename and url keys
        """
        if chat_id not in self._chat_sources:
            self._chat_sources[chat_id] = []
        
        # Add unique sources only (compare by URL)
        existing_sources = self._chat_sources[chat_id]
        existing_urls = set(source.get('url', '') for source in existing_sources)
        
        new_sources = [source for source in sources 
                       if source.get('url') not in existing_urls]
        
        if new_sources:
            self._chat_sources[chat_id].extend(new_sources)
            logger.debug(f"[{chat_id}] Added {len(new_sources)} new sources")
    
    def get_last_tool_calls(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get the most recent tool calls for a chat.
        
        Args:
            chat_id: Unique identifier for the chat
            
        Returns:
            The list of tool calls for the chat, or empty list if none found
        """
        return self._chat_tool_calls.get(chat_id, [])
    
    def get_last_tool_results(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get the most recent tool results for a chat.
        
        Args:
            chat_id: Unique identifier for the chat
            
        Returns:
            The list of tool results for the chat, or empty list if none found
        """
        return self._chat_tool_results.get(chat_id, [])
    
    def get_last_sources(self, chat_id: str) -> List[Dict[str, str]]:
        """Get the most recent sources for a chat.
        
        Args:
            chat_id: Unique identifier for the chat
            
        Returns:
            The list of sources for the chat, or empty list if none found
        """
        return self._chat_sources.get(chat_id, [])
    
    def clear_chat_data(self, chat_id: str):
        """Clear all data for a specific chat.
        
        Args:
            chat_id: Unique identifier for the chat to clear data for
        """
        if chat_id in self._chat_tool_calls:
            del self._chat_tool_calls[chat_id]
        if chat_id in self._chat_tool_results:
            del self._chat_tool_results[chat_id]
        if chat_id in self._chat_sources:
            del self._chat_sources[chat_id]
        logger.debug(f"[{chat_id}] Cleared all chat data") 