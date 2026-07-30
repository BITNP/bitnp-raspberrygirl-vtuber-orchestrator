"""模块契约说明.

职责: 提供 orchestrator.identity
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from dataclasses import dataclass
from typing import NewType, Protocol, final, override

VoiceProfileId = NewType("VoiceProfileId", str)

RecognitionConfidence = NewType("RecognitionConfidence", int)


@dataclass(frozen=True, slots=True)
class EncryptedVoiceTemplate:
    """类契约说明.

    职责: 保存 EncryptedVoiceTemplate
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: ciphertext。
    """

    ciphertext: bytes


class VoiceProfileVault(Protocol):
    """类契约说明.

    职责: 声明 VoiceProfileVault
    协议接口,约束实现方必须提供的行为。
    契约: 方法: store_encrypted、delete。
    """

    def store_encrypted(
        self,
        profile_id: VoiceProfileId,
        template: EncryptedVoiceTemplate,
    ) -> None:
        """函数契约说明.

        功能: 执行 store_encrypted
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。 template:
        EncryptedVoiceTemplate。 必填。
        契约: 同步调用。 返回 `None`。
        """
        ...

    def delete(self, profile_id: VoiceProfileId) -> None:
        """函数契约说明.

        功能: 执行 delete 的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。
        契约: 同步调用。 返回 `None`。
        """
        ...


@final
class InMemoryVoiceProfileVault:
    """类契约说明.

    职责: 定义 InMemoryVoiceProfileVault
    的状态、行为和对外协作边界。
    契约: 方法: __init__、store_encrypted、del
    ete、template。
    """

    def __init__(self) -> None:
        """函数契约说明.

        功能: 初始化
        InMemoryVoiceProfileVault
        的字段并建立实例不变式。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `None`。
        """
        self._templates: dict[VoiceProfileId, EncryptedVoiceTemplate] = {}

    def store_encrypted(
        self,
        profile_id: VoiceProfileId,
        template: EncryptedVoiceTemplate,
    ) -> None:
        """函数契约说明.

        功能: 执行 store_encrypted
        的同步逻辑,并维持签名契约。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。 template:
        EncryptedVoiceTemplate。 必填。
        契约: 同步调用。 返回 `None`。
        """
        self._templates[profile_id] = template

    def delete(self, profile_id: VoiceProfileId) -> None:
        """函数契约说明.

        功能: 执行 delete 的同步逻辑,并协调 pop。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。
        契约: 同步调用。 返回 `None`。
        """
        _ = self._templates.pop(profile_id, None)

    def template(self, profile_id: VoiceProfileId) -> EncryptedVoiceTemplate | None:
        """函数契约说明.

        功能: 执行 template 的同步逻辑,并协调 get。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。
        契约: 同步调用。 返回
        `EncryptedVoiceTemplate | None`。
        """
        return self._templates.get(profile_id)


@dataclass(frozen=True, slots=True)
class ProfileEnrollment:
    """类契约说明.

    职责: 保存 ProfileEnrollment
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: profile_id、preferred_name、en
    crypted_template、consented、confirmed
    、expires_at_ms。
    """

    profile_id: VoiceProfileId

    preferred_name: str

    encrypted_template: EncryptedVoiceTemplate

    consented: bool

    confirmed: bool = True

    expires_at_ms: int | None = None

    purpose: str = "personalization"


@dataclass(frozen=True, slots=True)
class ProfileRecognition:
    """类契约说明.

    职责: 保存 ProfileRecognition
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: profile_id、confidence。
    """

    profile_id: VoiceProfileId | None

    confidence: RecognitionConfidence


@dataclass(frozen=True, slots=True)
class ProfileCorrection:
    """类契约说明.

    职责: 保存 ProfileCorrection
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: profile_id、preferred_name。
    """

    profile_id: VoiceProfileId

    preferred_name: str


@dataclass(frozen=True, slots=True)
class ProfileRecognitionKnown:
    """类契约说明.

    职责: 保存 ProfileRecognitionKnown
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: profile_id、preferred_name。
    """

    profile_id: VoiceProfileId

    preferred_name: str


@dataclass(frozen=True, slots=True)
class ProfileRecognitionUnknown:
    """类契约说明.

    职责: 保存 ProfileRecognitionUnknown
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段、不变式和资源归属由类体声明与类型标注共同约束。
    """


type ProfileRecognitionResult = ProfileRecognitionKnown | ProfileRecognitionUnknown


@dataclass(frozen=True, slots=True)
class VoiceProfileConsentError(ValueError):
    """类契约说明.

    职责: 保存 VoiceProfileConsentError
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: profile_id。 方法: __str__。
    """

    profile_id: VoiceProfileId

    @override
    def __str__(self) -> str:
        """函数契约说明.

        功能: 生成面向日志、错误或调试输出的稳定文本表示。
        参数: self 表示当前实例。
        契约: 同步调用。 返回 `str`。
        """
        return f"voice profile enrollment requires consent: {self.profile_id}"
