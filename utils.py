"""
Utility functions for the Legal Mind Map Generator
"""
import re
import json
import logging
import uuid

logger = logging.getLogger(__name__)

def clean_and_parse_json(json_text):
    """
    Clean and parse JSON text with better error handling and cleanup.
    
    Args:
        json_text (str): Raw JSON text that may contain syntax errors
        
    Returns:
        dict: Parsed JSON object or error structure
    """
    try:
        # First, try to parse as-is
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parse failed: {e}")
        logger.debug(f"Error at position {e.pos}")
        
        # Try to clean common issues
        cleaned_text = json_text
        
        # Fix common bracket/parentheses issues
        # Remove double closing brackets in arrays
        cleaned_text = re.sub(r'\]\](?=\s*[,}])', ']', cleaned_text)
        
        # Fix trailing commas before closing brackets/braces
        cleaned_text = re.sub(r',(\s*[}\]])', r'\1', cleaned_text)
        
        # Fix missing commas between array elements (basic case)
        cleaned_text = re.sub(r'"\s*\n\s*"', '",\n"', cleaned_text)
        
        # Fix specific known issues
        if 'testimonies"]]' in cleaned_text:
            cleaned_text = cleaned_text.replace('testimonies"]]', 'testimonies"]')
        
        # Try parsing again
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e2:
            logger.error(f"Cleaned JSON parse also failed: {e2}")
            logger.debug(f"Error at position {e2.pos}")
            
            # Show the problematic area for debugging
            start = max(0, e2.pos - 50)
            end = min(len(cleaned_text), e2.pos + 50)
            problematic_area = cleaned_text[start:end]
            logger.debug(f"Problematic area: ...{problematic_area}...")
            
            # If all else fails, return a basic structure with error info
            return {
                "nodes": [],
                "edges": [],
                "error": f"Failed to parse JSON: {str(e2)}",
                "original_error_pos": e2.pos
            }

def validate_graph_structure(graph_data):
    """More flexible validation allowing partial edge data"""
    if not isinstance(graph_data, dict):
        return False
    
    if "nodes" not in graph_data or "edges" not in graph_data:
        return False
    
    if not isinstance(graph_data["nodes"], list) or not isinstance(graph_data["edges"], list):
        return False
    
    # Validate node structure
    for node in graph_data["nodes"]:
        if not isinstance(node, dict):
            return False
        if "id" not in node or "type" not in node or "label" not in node:
            return False
    
    return True  # Edges can be validated during sanitization


def sanitize_graph_data(graph_data):
    """Enhanced sanitization to handle invalid edges"""
    if not validate_graph_structure(graph_data):
        logger.warning("Invalid graph structure detected, returning empty graph")
        return {"nodes": [], "edges": []}
    
    # Clean nodes
    seen_node_ids = set()
    unique_nodes = []
    for node in graph_data["nodes"]:
        if not isinstance(node, dict): 
            continue
            
        # Ensure required fields exist
        node.setdefault("details", "")
        node.setdefault("place", "")
        node.setdefault("date", "")
        node.setdefault("amount", "")
        node.setdefault("actions", [])
        node.setdefault("case_references", [])
        
        if node["id"] not in seen_node_ids:
            unique_nodes.append(node)
            seen_node_ids.add(node["id"])
    
    # Clean edges - accept partial data
    valid_edges = []
    valid_node_ids = {node["id"] for node in unique_nodes}
    
    for edge in graph_data["edges"]:
        if not isinstance(edge, dict):
            continue
            
        # Ensure required fields with defaults
        edge.setdefault("id", str(uuid.uuid4()))
        edge.setdefault("source", "")
        edge.setdefault("target", "")
        edge.setdefault("label", "related_to")
        
        # Only keep edges with valid references
        if (edge["source"] in valid_node_ids and 
            edge["target"] in valid_node_ids):
            valid_edges.append(edge)
    
    return {
        "nodes": unique_nodes,
        "edges": valid_edges
    }