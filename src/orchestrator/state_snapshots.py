
from dataclasses import dataclass
from typing import NewType

MemoryRevision = NewType("MemoryRevision", int)

ContextGeneration = NewType("ContextGeneration", int)

ProfileRevision = NewType("ProfileRevision", int)

ConsentRevision = NewType("ConsentRevision", int)

CorpusRevision = NewType("CorpusRevision", int)

IndexRevision = NewType("IndexRevision", int)


@dataclass(frozen=True, slots=True)
class TaskStateSnapshot:

    memory_revision: MemoryRevision

    context_generation: ContextGeneration

    profile_revision: ProfileRevision

    consent_revision: ConsentRevision

    corpus_revision: CorpusRevision

    index_revision: IndexRevision

    @classmethod
    def initial(cls) -> "TaskStateSnapshot":
        return cls(
            memory_revision=MemoryRevision(0),
            context_generation=ContextGeneration(0),
            profile_revision=ProfileRevision(0),
            consent_revision=ConsentRevision(0),
            corpus_revision=CorpusRevision(0),
            index_revision=IndexRevision(0),
        )
