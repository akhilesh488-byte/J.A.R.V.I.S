"""
Shim module pointing to core.prompt.
Provides access to load_prompt_templates and build_system_prompt.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.prompt import *
