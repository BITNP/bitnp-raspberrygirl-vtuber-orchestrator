"""模块契约说明.

职责: 提供 orchestrator.state_snapshots
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

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
    """类契约说明.

    职责: 保存 TaskStateSnapshot
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: memory_revision、context_gene
    ration、profile_revision、consent_revi
    sion、corpus_revision、index_revision。
    方法: initial。
    """

    memory_revision: MemoryRevision

    context_generation: ContextGeneration

    profile_revision: ProfileRevision

    consent_revision: ConsentRevision

    corpus_revision: CorpusRevision

    index_revision: IndexRevision

    @classmethod
    def initial(cls) -> "TaskStateSnapshot":
        """函数契约说明.

        功能: 执行 initial 的同步逻辑,并协调 cls,
        MemoryRevision,
        ContextGeneration,
        ProfileRevision。
        参数: cls 表示当前类。
        契约: 同步调用。 返回
        `'TaskStateSnapshot'`。
        """
        return cls(
            memory_revision=MemoryRevision(0),
            context_generation=ContextGeneration(0),
            profile_revision=ProfileRevision(0),
            consent_revision=ConsentRevision(0),
            corpus_revision=CorpusRevision(0),
            index_revision=IndexRevision(0),
        )
