"""Decides which entity a statement refers to, and whether a candidate fact may be stored.

Fixed rules decide, never a language model. A model may only propose candidates."""
from .entity_resolver import EntityResolver, Resolution
from .extraction import STRUCTURED_PROMPT, parse_reply
from .fact_validator import ACCEPTED, REJECTED, UNCERTAIN, Candidate, Decision, FactValidator, KnowledgePipeline, SourceLine
from .conflict import Conflict
from .state_engine import Period, SlotState, StateEngine
from .evidence_assembler import EvidenceAssembler
from .knowledge_retriever import KnowledgeRetriever
from .query_router import QueryRouter
