#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""

    # --- Add src directory to Python path ---
    # Get the directory containing manage.py (project root)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    # Path to the src directory
    SRC_PATH = os.path.join(PROJECT_ROOT, "src")
    # Add src to the beginning of sys.path if it's not already there
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)
        print(f"--- Added '{SRC_PATH}' to sys.path ---")  # Optional: confirmation log
    # --- End Add src directory ---

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ipl_django_project.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
