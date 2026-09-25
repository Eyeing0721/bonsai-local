"""本地知识库：把用户自己的资料变成模型能查到的东西。

为什么这个模型需要它 —— 这不是猜的，是实测出来的。Ternary-Bonsai-2-27B 是
1.75 bpw 的三值量化，我们拿 31 个"懂行的人必须提到的点"去考它（红队方向），
挂上最好的那个适配器也只命中 35.5%，而 SeImpersonate、tcache、AmsiScanBuffer
这类关键机制名五个配置一次都没出现。反过来，它却会编出 "CVE-2023-38816
(PrintNightmare 2.0)" 这种不存在的编号。适配器能改的是行为倾向，改不了知识
存量 —— 缺的那块只能靠检索补。

为什么要两条召回路径而不是一条：
  BM25      纯 Python，零依赖，零下载。CVE 号、函数名、寄存器名这类"精确词"
            是它最强的地方 —— 而精确词恰好就是模型最缺的那部分。
  向量检索  需要额外的向量模型（610 MB，按需下载）。同义改写、跨语言提问、
            "意思相近但用词不同"的场景它更强。
用户真正会问的东西一半是精确词（"CVE-2018-8120 是哪个组件的"），一半是语义
（"有没有讲过提权的资料"），所以两条都要。而 BM25 永远可用、向量可有可无：
向量模型没下载、下载失败、或者显存不够，知识库依然工作，只是差一点。
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings

# 切块大小按"字符"而不是 token 算：中文一个字大约一个 token，英文一个词大约
# 一到两个，所以按字符切对中英混排最省心，而且不需要为了切块先加载分词器。
CHUNK_CHARS = 420
CHUNK_OVERLAP = 80

# 检索结果的默认条数与拼进上下文的预算。3200 字大约 2000-2500 token，
# 对 16k 上下文来说留足了对话空间。
TOP_K = 6
CONTEXT_BUDGET = 3200

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".java", ".kt", ".c", ".h",
    ".cpp", ".hpp", ".cc", ".cs", ".go", ".rs", ".rb", ".php", ".swift",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".sql",
    ".html", ".htm", ".css", ".scss", ".xml", ".json", ".jsonl", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".properties", ".gradle",
}

MAX_FILE_BYTES = 64 << 20          # 单个文件 64 MB 上限，再大基本是误选


# ------------------------------------------------------------------ 分词
_ASCII_WORD = re.compile(r"[a-z0-9_]+")
_CJK_RUN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]+")


def tokenize(text: str) -> list[str]:
    """BM25 用的词元。

    中文不引分词器（那会多一个依赖和几 MB 词典），改用**字符二元组**：这是
    CJK 检索里用了很多年的做法，效果接近分词而完全不需要词典。英文走常规
    小写词元。CVE 号、`0xC2B2AE35` 这类带连字符和数字的串会被拆成几段，
    但这不影响 —— 查询里同样会被拆成一样的几段。
    """
    low = text.lower()
    toks = _ASCII_WORD.findall(low)
    for run in _CJK_RUN.findall(low):
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    return toks


# ------------------------------------------------------------------ 切块
def _atoms(text: str, size: int) -> list[str]:
    """先按空行切段，过长的段再硬切。"""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            out.append(para)
            continue
        step = max(1, size - CHUNK_OVERLAP)
        for i in range(0, len(para), step):
            piece = para[i:i + size].strip()
            if piece:
                out.append(piece)
    return out


def chunk_text(text: str, size: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """把长文切成有重叠的块。

    重叠是为了不让答案正好卡在切口上 —— 只切不叠的话，一个跨块的结论会两边
    都不完整，检索命中哪一块都答不全。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks: list[str] = []
    buf = ""
    for a in _atoms(text, size):
        if not buf:
            buf = a
        elif len(buf) + 2 + len(a) <= size:
            buf += "\n\n" + a
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + " " + a) if tail else a
        while len(buf) > size:              # 硬切过的块加上尾巴可能仍超长
            chunks.append(buf[:size])
            buf = buf[size - overlap:]
    if buf.strip():
        chunks.append(buf)
    return [c for c in chunks if len(c.strip()) >= 8]


# ------------------------------------------------------------------ BM25
class BM25:
    """Okapi BM25，带倒排表。

    倒排表不只是为了快，也是为了内存：每个查询词只碰真正含有它的那些块，
    而不是拿查询去扫全部块。
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5,
                 b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        for i, toks in enumerate(corpus):
            self.doc_len.append(len(toks))
            for term, cnt in Counter(toks).items():
                self.postings[term].append((i, cnt))
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 1.0
        self.idf: dict[str, float] = {
            t: math.log(1.0 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        } if self.n else {}

    def score(self, query_tokens: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        if not self.n:
            return out
        for term in set(query_tokens):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for idx, freq in posting:
                dl = self.doc_len[idx] or 1
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                out[idx] += idf * freq * (self.k1 + 1) / denom
        return out


# ------------------------------------------------------------------ 记录
@dataclass
class Doc:
    id: str
    name: str
    path: str
    chars: int
    chunks: int
    added: float

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "path": self.path,
                "chars": self.chars, "chunks": self.chunks, "added": self.added}


@dataclass
class Hit:
    text: str
    doc_name: str
    score: float
    chunk_id: int
    matched: str = ""            # 命中的是关键词还是语义，界面用来做解释

    def as_dict(self) -> dict:
        return {"text": self.text, "doc": self.doc_name, "score": self.score}


# ------------------------------------------------------------------ 读写文件
def _decode(raw: bytes) -> str:
    """中文文本文件编码很杂，按常见顺序试。

    先 utf-8（带 BOM 的也认），再 gb18030（GBK/GB2312 的超集），最后 latin-1
    兜底 —— 兜底永远不会抛异常，宁可出现几个乱码字符也不要整个文件读不了。
    """
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader           # 可选依赖
        except ImportError as e:
            raise RuntimeError("读 PDF 需要 pypdf，请先 pip install pypdf") from e
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext in (".docx", ".doc"):
        raise RuntimeError("暂不支持 Word 文档，请先另存为 txt 或 PDF")
    if ext in TEXT_EXTS or not ext:
        return _decode(path.read_bytes())
    raise RuntimeError(f"不认识的文件类型 {ext}")


# ------------------------------------------------------------------ 主类
class Knowledge:
    """知识库的持久化 + 检索。

    磁盘布局（都在 <数据目录>/knowledge/ 下）：
      index.json    文档清单和向量维度
      chunks.jsonl  每行一个块，检索时全量载入内存
      vectors.npy   块向量，只有在启用向量检索之后才存在

    为什么不用真正的向量数据库：个人知识库的量级是几千到几万个块，一个
    float32 矩阵加一次点积就是全部所需。引一个向量库进来，发布包要大几 MB、
    还会多出一个需要维护的进程，换不到任何东西。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self.docs: list[Doc] = []
        self.texts: list[str] = []
        self.doc_of: list[str] = []          # 块 -> 文档 id
        self.vectors = None                  # np.ndarray | None
        self.dim = 0
        self._bm25: BM25 | None = None
        self._tokens: list[list[str]] | None = None
        self.loaded = False

    # -- 路径 ----------------------------------------------------------
    @property
    def root(self) -> Path:
        return self.settings.data_dir / "knowledge"

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    @property
    def chunks_path(self) -> Path:
        return self.root / "chunks.jsonl"

    @property
    def vectors_path(self) -> Path:
        return self.root / "vectors.npy"

    # -- 载入 / 保存 ---------------------------------------------------
    def load(self) -> None:
        with self._lock:
            self.docs, self.texts, self.doc_of = [], [], []
            self.vectors, self.dim = None, 0
            self._bm25, self._tokens = None, None
            try:
                meta = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.dim = int(meta.get("dim") or 0)
                self.docs = [Doc(**d) for d in meta.get("docs", [])]
            except Exception:                                   # noqa: BLE001
                self.docs, self.dim = [], 0
            if self.chunks_path.exists():
                with self.chunks_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:                           # noqa: BLE001
                            continue
                        self.texts.append(rec.get("text") or "")
                        self.doc_of.append(rec.get("doc") or "")
            if self.dim and self.vectors_path.exists():
                try:
                    import numpy as np
                    arr = np.load(self.vectors_path)
                    if arr.shape[0] == len(self.texts):
                        self.vectors = arr
                except Exception:                                   # noqa: BLE001
                    self.vectors = None
                    self.dim = 0
            self.loaded = True

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"version": 1, "dim": self.dim,
             "docs": [d.as_dict() for d in self.docs]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.index_path)

        tmp = self.chunks_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for text, doc in zip(self.texts, self.doc_of):
                fh.write(json.dumps({"doc": doc, "text": text},
                                    ensure_ascii=False) + "\n")
        os.replace(tmp, self.chunks_path)

        if self.vectors is not None and self.dim:
            import numpy as np
            tmpv = self.vectors_path.with_suffix(".npy.tmp")
            with tmpv.open("wb") as fh:
                np.save(fh, self.vectors)
            os.replace(tmpv, self.vectors_path)
        elif self.vectors_path.exists():
            self.vectors_path.unlink(missing_ok=True)

    # -- 索引 ----------------------------------------------------------
    def _rebuild(self) -> None:
        self._tokens = [tokenize(t) for t in self.texts]
        self._bm25 = BM25(self._tokens)

    def _ensure_index(self) -> None:
        if self._bm25 is None:
            self._rebuild()

    # -- 增删 ----------------------------------------------------------
    def add_file(self, path: Path, embed=None, progress=None) -> Doc:
        """读一个文件、切块、入索引。`embed` 是 embed_fn(texts)->vectors。"""
        path = Path(path)
        if not path.is_file():
            raise RuntimeError("选中的不是文件")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise RuntimeError("文件超过 64 MB，请先拆分")
        with self._lock:
            text = extract_text(path)
            if not text.strip():
                raise RuntimeError("这个文件里没有可读的文字（可能是扫描件图片）")
            chunks = chunk_text(text)
            if not chunks:
                raise RuntimeError("这个文件太短或切不出有效的段落")

            doc = Doc(id=_new_doc_id(self.docs), name=path.name, path=str(path),
                      chars=len(text), chunks=len(chunks), added=_now())
            new_vecs = None
            if embed is not None and self.dim:
                if progress is not None:
                    progress.set(stage="extracting", label=f"索引「{path.name}」",
                                 done=0, total=len(chunks), detail="")
                new_vecs = embed(chunks)

            self.docs.append(doc)
            self.texts.extend(chunks)
            self.doc_of.extend([doc.id] * len(chunks))
            if new_vecs is not None:
                self._append_vectors(new_vecs)
            self._rebuild()
            self._save()
            return doc

    def add_dir(self, folder: Path, embed=None, progress=None,
                recursive: bool = True) -> dict:
        folder = Path(folder)
        if not folder.is_dir():
            raise RuntimeError("选中的不是文件夹")
        pattern = "**/*" if recursive else "*"
        files = sorted(p for p in folder.glob(pattern) if p.is_file())
        added, skipped, failed = [], 0, []
        for p in files:
            ext = p.suffix.lower()
            if ext not in TEXT_EXTS and ext != ".pdf":
                skipped += 1
                continue
            try:
                added.append(self.add_file(p, embed=embed, progress=progress))
            except Exception as e:                              # noqa: BLE001
                failed.append(f"{p.name}: {e}")
        return {"added": [d.as_dict() for d in added], "skipped": skipped,
                "failed": failed[:20], "scanned": len(files)}

    def remove(self, doc_id: str) -> None:
        with self._lock:
            keep = [i for i, d in enumerate(self.doc_of) if d != doc_id]
            drop = set(range(len(self.doc_of))) - set(keep)
            self.texts = [self.texts[i] for i in keep]
            self.doc_of = [self.doc_of[i] for i in keep]
            self.docs = [d for d in self.docs if d.id != doc_id]
            if self.vectors is not None and drop:
                import numpy as np
                self.vectors = self.vectors[sorted(keep)] if keep else None
            self._rebuild()
            self._save()

    def clear(self) -> None:
        with self._lock:
            self.docs, self.texts, self.doc_of = [], [], []
            self.vectors, self.dim = None, 0
            self._rebuild()
            self._save()

    # -- 向量 ----------------------------------------------------------
    def _append_vectors(self, vecs) -> None:
        import numpy as np
        arr = np.asarray(vecs, dtype="float32")
        if self.vectors is None:
            self.vectors = arr
        else:
            self.vectors = np.vstack([self.vectors, arr])
        self.dim = int(self.vectors.shape[1])

    def reindex_vectors(self, embed, progress=None) -> int:
        """全部重算向量。启用向量检索、或者换了向量模型之后调用。"""
        with self._lock:
            if not self.texts:
                self.vectors, self.dim = None, 0
                self._save()
                return 0
            if progress is not None:
                progress.set(stage="extracting", label="建立向量索引",
                             done=0, total=len(self.texts), detail="")
            vecs = embed(self.texts)
            self._append_vectors(vecs)
            self._save()
            return len(self.texts)

    # -- 检索 ----------------------------------------------------------
    def search(self, query: str, k: int = TOP_K, embed=None) -> list[Hit]:
        with self._lock:
            if not self.texts:
                return []
            self._ensure_index()
            assert self._bm25 is not None

            # 两条路径各出一个排名，再用 RRF 融合。用排名而不是原始分，是因为
            # BM25 的分数没有上界、余弦在 [-1,1]，两者量纲不可比，硬加权只会
            # 让其中一个永远压住另一个。
            rankings: list[list[int]] = []
            lex = self._bm25.score(tokenize(query))
            if lex:
                rankings.append([i for i, _ in
                                 sorted(lex.items(), key=lambda kv: -kv[1])[:60]])
            dense_hit = False
            if embed is not None and self.vectors is not None and self.dim:
                try:
                    import numpy as np
                    qv = np.asarray(embed([query])[0], dtype="float32")
                    norm = float(np.linalg.norm(qv))
                    if norm > 0:
                        sims = self.vectors @ (qv / norm)
                        order = np.argsort(-sims)[:60]
                        rankings.append([int(i) for i in order])
                        dense_hit = True
                except Exception:                               # noqa: BLE001
                    dense_hit = False

            if not rankings:
                return []

            fused: dict[int, float] = defaultdict(float)
            for ranking in rankings:
                for rank, idx in enumerate(ranking):
                    fused[idx] += 1.0 / (60 + rank + 1)

            hits: list[Hit] = []
            names = {d.id: d.name for d in self.docs}
            for idx, score in sorted(fused.items(), key=lambda kv: -kv[1])[:k]:
                if idx >= len(self.texts):
                    continue
                hits.append(Hit(text=self.texts[idx],
                                doc_name=names.get(self.doc_of[idx], "资料"),
                                score=round(float(score), 6), chunk_id=idx,
                                matched="语义+关键词" if dense_hit else "关键词"))
            return hits

    # -- 给界面 --------------------------------------------------------
    def state(self, dense_ready: bool = False) -> dict:
        with self._lock:
            return {
                "docs": [d.as_dict() for d in self.docs],
                "chunks": len(self.texts),
                "chars": sum(d.chars for d in self.docs),
                "dense": self.dim > 0 and self.vectors is not None,
                "dense_ready": dense_ready,
                "enabled": bool(self.settings.get("kb_enabled", True)),
            }

    def build_context(self, query: str, k: int = TOP_K, embed=None,
                      budget: int = CONTEXT_BUDGET) -> tuple[str, list[Hit]]:
        """把命中的块拼成给模型看的参考资料。返回 (文本, 命中列表)。"""
        hits = self.search(query, k=k, embed=embed)
        if not hits:
            return "", []
        parts: list[str] = []
        used = 0
        kept: list[Hit] = []
        for i, h in enumerate(hits, 1):
            piece = f"【资料 {i}｜{h.doc_name}】\n{h.text.strip()}"
            if used + len(piece) > budget and kept:
                break
            parts.append(piece)
            kept.append(h)
            used += len(piece)
        return "\n\n".join(parts), kept


# ------------------------------------------------------------------ 小工具
def _now() -> float:
    import time
    return time.time()


def _new_doc_id(existing: list[Doc]) -> str:
    import secrets
    used = {d.id for d in existing}
    while True:
        candidate = secrets.token_hex(4)
        if candidate not in used:
            return candidate


# ------------------------------------------------------------------ 提示词
SYSTEM_TEMPLATE = """下面是从用户自己的资料里检索到的片段。回答时优先依据它们，\
并指出依据来自哪份资料。如果这些片段不足以回答问题，就直接说资料里没有，\
不要凭印象补充 —— 你补出来的细节很可能是错的。

{context}
"""


def build_messages(messages: list[dict], context: str) -> list[dict]:
    """把资料插成一条 system 消息。

    插在最前面而不是塞进用户那条消息里：一来不污染用户看到的原文，二来多轮
    对话里 system 的位置固定，模型更容易稳定地把它当成背景而不是当成当前
    指令。
    """
    if not context:
        return messages
    note = {"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)}
    out = []
    inserted = False
    for m in messages:
        if not inserted and m.get("role") == "system":
            out.append(m)
            out.append(note)
            inserted = True
            continue
        out.append(m)
    if not inserted:
        out.insert(0, note)
    return out
