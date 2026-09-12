import asyncio
from pathlib import Path

import pytest
from llama_index.core.node_parser.text import sentence

from orchestrator.knowledge_corpus import CorpusLoadError
from orchestrator.modes import AnswerCandidate, AudienceInput, AudienceSource
from orchestrator.retrieval import ReadonlyCorpusConfig, ReadonlyLlamaIndexProvider


def _query(text: str) -> AnswerCandidate:
    return AnswerCandidate(AudienceInput(AudienceSource.ASR, text, 1))


def test_chinese_keyword_ranking_and_no_match(tmp_path: Path) -> None:
    _ = (tmp_path / "catering.txt").write_text("食堂供应午餐和晚餐", encoding="utf-8")
    _ = (tmp_path / "product.md").write_text(
        "树莓女孩支持语音识别、幻灯片演示", encoding="utf-8"
    )
    _ = (tmp_path / "schedule.json").write_text(
        '{"展会时间":"周六上午十点"}', encoding="utf-8"
    )
    _ = (tmp_path / "ignored.bin").write_bytes(b"ignored")
    provider = ReadonlyLlamaIndexProvider(ReadonlyCorpusConfig(tmp_path, top_k=1))

    result = asyncio.run(provider.retrieve_async(_query("幻灯片演示")))

    assert len(result.refs) == 1
    assert result.refs[0].ref_id == "product.md#0"
    assert result.refs[0].corpus_revision == provider.snapshot.corpus_revision
    assert provider.retrieve(_query("展会时间")).refs[0].ref_id == "schedule.json#0"
    assert provider.retrieve(_query("quantumxyz")).refs == ()


def test_corpus_snapshot_is_frozen_and_reload_changes_revision(tmp_path: Path) -> None:
    path = tmp_path / "product.md"
    _ = path.write_text("展会周六举行", encoding="utf-8")
    first = ReadonlyLlamaIndexProvider(ReadonlyCorpusConfig(tmp_path))
    same = ReadonlyLlamaIndexProvider(ReadonlyCorpusConfig(tmp_path))
    assert first.snapshot == same.snapshot
    _ = path.write_text("展会周日举行", encoding="utf-8")
    second = ReadonlyLlamaIndexProvider(ReadonlyCorpusConfig(tmp_path))
    assert first.snapshot != second.snapshot
    assert "周六" in first.retrieve(_query("展会")).refs[0].text
    assert "周日" in second.retrieve(_query("展会")).refs[0].text


def test_empty_directory_is_a_valid_empty_knowledge_snapshot(tmp_path: Path) -> None:
    provider = ReadonlyLlamaIndexProvider(ReadonlyCorpusConfig(tmp_path))
    assert provider.retrieve(_query("介绍产品")).refs == ()


def test_corpus_rejects_symlink_escape(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    outside = tmp_path / "private.txt"
    _ = outside.write_text("不能读入的资料", encoding="utf-8")
    (corpus / "escape.txt").symlink_to(outside)
    with pytest.raises(CorpusLoadError):
        _ = ReadonlyLlamaIndexProvider(ReadonlyCorpusConfig(corpus))


def test_corpus_rejects_oversized_file(tmp_path: Path) -> None:
    _ = (tmp_path / "oversized.txt").write_bytes(b"x" * 1_048_577)
    with pytest.raises(CorpusLoadError):
        _ = ReadonlyLlamaIndexProvider(ReadonlyCorpusConfig(tmp_path))


def test_retrieval_cooperatively_cancels_without_worker_threads(tmp_path: Path) -> None:
    _ = (tmp_path / "product.txt").write_text("幻灯片演示", encoding="utf-8")
    provider = ReadonlyLlamaIndexProvider(ReadonlyCorpusConfig(tmp_path))

    async def scenario() -> None:
        task = asyncio.create_task(provider.retrieve_async(_query("演示")))
        await asyncio.sleep(0)
        assert not task.done()
        _ = task.cancel()
        with pytest.raises(asyncio.CancelledError):
            _ = await task

    asyncio.run(scenario())


def test_knowledge_ingestion_never_initializes_network_tokenizers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbidden_tokenizer() -> None:
        pytest.fail("offline knowledge must not initialize tiktoken or NLTK")

    monkeypatch.setattr(sentence, "get_tokenizer", forbidden_tokenizer)
    monkeypatch.setattr(sentence, "split_by_sentence_tokenizer", forbidden_tokenizer)
    _ = (tmp_path / "product.md").write_text(
        "树莓女孩。支持幻灯片演示\uff01" * 100, encoding="utf-8"
    )
    provider = ReadonlyLlamaIndexProvider(ReadonlyCorpusConfig(tmp_path))
    assert provider.retrieve(_query("幻灯片演示")).refs
