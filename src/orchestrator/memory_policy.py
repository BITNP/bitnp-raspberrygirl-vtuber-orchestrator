"""Deterministic boundaries shared by proposal admission and durable reload."""

import re
import unicodedata

MAX_MEMORY_PROVENANCE_BYTES = 256
MAX_MEMORY_CONFIDENCE = 100
MAX_MEMORY_ENTRIES = 32
MAX_MEMORY_TEXT_BYTES = 16_384
MAX_MEMORY_DOCUMENT_BYTES = 262_144
MAX_MEMORY_AUDIT_RECORDS = 64
MAX_MEMORY_CONFLICT_RECORDS = 16
MAX_MEMORY_KEY_CHARS = 128
MAX_MEMORY_VALUE_CHARS = 512

_KEY = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
# Defense in depth, independent of the extractor's self-declared category.
# This intentionally rejects ambiguous sensitive material, including discussion
# of these categories. It is not a general semantic privacy classifier.
_SENSITIVE_PATTERNS = (
    r"健康|疾病|患有|诊断|病史|病历|糖尿病|抑郁|焦虑症|癌症|用药|药物|过敏|怀孕",
    r"财务|收入|薪资|工资|存款|负债|银行卡|信用卡|账户余额|资产",
    r"政治|政党|党派|投票|宗教|信仰|性取向|性生活",
    r"联系方式|邮箱|邮件地址|手机号|电话号码|家庭住址|身份证|护照",
    r"密码|口令|凭据|密钥|令牌|访问令牌|私钥|声纹|指纹|生物特征|嵌入向量",
    r"health|medical|diagnos|disease|diabet|depress|medication|allerg|pregnan",
    r"financ|salary|income|debt|bank.?account|credit.?card",
    r"politic|party.?affiliation|religio|sexual",
    r"e.?mail|phone|address|passport|social.?security",
    r"password|passwd|credential|secret|token|api.?key|private.?key",
    r"biometric|voiceprint|fingerprint|embedding",
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}",
    r"(?:\d[\s().+-]*){7,}|-----begin",
)
_SENSITIVE = re.compile("|".join(_SENSITIVE_PATTERNS), re.IGNORECASE)


def contains_sensitive_memory(key: str, value: str) -> bool:
    text = unicodedata.normalize("NFKC", f"{key} {value}")
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    return _SENSITIVE.search(text) is not None


def valid_memory_text(key: str, value: str) -> bool:
    return (
        len(key) <= MAX_MEMORY_KEY_CHARS
        and _KEY.fullmatch(key) is not None
        and 0 < len(value) <= MAX_MEMORY_VALUE_CHARS
        and value == value.strip()
        and all(unicodedata.category(char)[0] != "C" for char in value)
        and not any(marker in value for marker in ("<!--", "-->", "<", ">"))
    )
