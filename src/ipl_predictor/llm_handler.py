import requests
import json
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:instruct")


def query_ollama_llm(prompt_text: str) -> str:
    """
    Sends a prompt to the Ollama LLM and returns its response.
    """
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False},
            timeout=60,  # Increased timeout for potentially longer queries
        )
        response.raise_for_status()
        return response.json().get("response", "Error: No response field from LLM.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Ollama request failed: {e}")
        return f"Error: Could not connect to LLM. Details: {str(e)}"
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to decode Ollama response: {e}")
        return "Error: Invalid response format from LLM."
    except Exception as e:
        print(f"ERROR: Unexpected error during Ollama call: {e}")
        return f"Error: An unexpected issue occurred with the LLM. Details: {str(e)}"
