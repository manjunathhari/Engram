"""Engram's knowledge layer: typed records traced to their sentences, which entity a statement means, whether a candidate may be stored,
how facts change over time, and answering structured questions. Fixed rules decide, never a language model.

Names load on first use, so the storage files can be imported by engram_chat without loading the rest.
"""
import importlib

_HOME = dict(
    models=("Entity", "EntityAlias", "KnowledgeFact", "FactEvidence", "FactWithEvidence", "SourceSentence", "RecordError", "normalize_name", "tidy"),
    schema=("SCHEMA", "SCHEMA_VERSION"), database=("KnowledgeBase", "prepare"), evidence=("forget_sentence",), migrations=("migrate",), serialization=("export_knowledge", "import_knowledge", "backup", "integrity_report"),
    entity_resolver=("EntityResolver", "Resolution"), extraction=("STRUCTURED_PROMPT", "parse_reply"),
    fact_validator=("ACCEPTED", "REJECTED", "UNCERTAIN", "Candidate", "Decision", "FactValidator", "KnowledgePipeline", "SourceLine"),
    conflict=("Conflict",), state_engine=("Period", "SlotState", "StateEngine"), evidence_assembler=("EvidenceAssembler",),
    knowledge_retriever=("KnowledgeRetriever",), query_router=("QueryRouter",))
_WHERE = {name: module for module, names in _HOME.items() for name in names}
__all__ = sorted(_WHERE)


def __getattr__(name):
    if name in _WHERE: return getattr(importlib.import_module(f".{_WHERE[name]}", __name__), name)
    raise AttributeError(name)
