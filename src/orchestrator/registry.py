"""模块契约说明.

职责: 提供 orchestrator.registry
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass
from typing import override

from orchestrator.ids import ConnectionId


@dataclass(frozen=True, slots=True, order=True)
class ModuleIdentity:
    """类契约说明.

    职责: 保存 ModuleIdentity
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: module_name、instance_id。 方法:
    parse、label。
    """

    module_name: str

    instance_id: str

    @classmethod
    def parse(cls, *, module_name: str, instance_id: str) -> "ModuleIdentity":
        """函数契约说明.

        功能: 从边界输入解析类型化值。
        参数: cls 表示当前类。 module_name: str。
        必填。 instance_id: str。 必填。
        契约: 同步调用。 返回 `'ModuleIdentity'`。
        可能抛出 ModuleIdentityParseError。
        """
        cleaned_module_name = module_name.strip()

        cleaned_instance_id = instance_id.strip()

        if cleaned_module_name == "":
            raise ModuleIdentityParseError(field_name="module_name")

        if cleaned_instance_id == "":
            raise ModuleIdentityParseError(field_name="instance_id")

        return cls(module_name=cleaned_module_name, instance_id=cleaned_instance_id)

    def label(self) -> str:
        """函数契约说明.

        功能: 执行 label 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"{self.module_name}/{self.instance_id}"


@dataclass(frozen=True, slots=True)
class RegisteredConnection:
    """类契约说明.

    职责: 保存 RegisteredConnection
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: identity、connection_id。
    """

    identity: ModuleIdentity

    connection_id: ConnectionId


@dataclass(frozen=True, slots=True)
class ModuleIdentityParseError(Exception):
    """类契约说明.

    职责: 保存 ModuleIdentityParseError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: field_name。 方法: __str__。
    """

    field_name: str

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"module identity field is blank: {self.field_name}"


@dataclass(frozen=True, slots=True)
class DuplicateModuleIdentityError(Exception):
    """类契约说明.

    职责: 保存 DuplicateModuleIdentityError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: identity。 方法: __str__。
    """

    identity: ModuleIdentity

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"active module identity already registered: {self.identity.label()}"


class ConnectionRegistry:
    """类契约说明.

    职责: 定义 ConnectionRegistry
    的状态、行为和对外协作边界。
    契约: 方法:
    __init__、register、active_identities。
    """

    def __init__(self) -> None:
        """函数契约说明.

        功能: 初始化 ConnectionRegistry
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        self._active: dict[ModuleIdentity, RegisteredConnection] = {}

    def register(
        self,
        identity: ModuleIdentity,
        connection_id: ConnectionId,
    ) -> RegisteredConnection:
        """函数契约说明.

        功能: 执行 register 的同步逻辑,并协调
        RegisteredConnection,
        DuplicateModuleIdentityError。
        参数: self 表示当前实例。 identity:
        ModuleIdentity。 必填。
        connection_id: ConnectionId。 必填。
        契约: 同步调用。 返回
        `RegisteredConnection`。 可能抛出
        DuplicateModuleIdentityError。
        """
        if identity in self._active:
            raise DuplicateModuleIdentityError(identity=identity)

        registered = RegisteredConnection(
            identity=identity,
            connection_id=connection_id,
        )

        self._active[identity] = registered

        return registered

    def active_identities(self) -> tuple[ModuleIdentity, ...]:
        """函数契约说明.

        功能: 执行 active_identities
        的同步逻辑,并协调 tuple, sorted。
        参数: self 表示当前实例。
        契约: 同步调用。 返回
        `tuple[ModuleIdentity, ...]`。
        """
        return tuple(sorted(self._active))
