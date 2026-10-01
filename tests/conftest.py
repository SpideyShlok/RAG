import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub torch/chroma/neo4j/ollama/etc. before any test module imports app code,
# since this suite runs without those services/models installed.
import conftest_stubs  # noqa: E402,F401
