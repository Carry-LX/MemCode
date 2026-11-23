# src/core/memory_structures.py

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class BaseMemoryUnit:
    """
    底层：原始对话单元。
    """
    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    source_document: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AtomicMemoryUnit:
    """Atomic fact with source-dialogue pointers and discrete VQ-VAE code IDs."""
    fact_text: str
    source_base_ids: List[str]

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    code_indices: List[int] = field(default_factory=list)  # VQ-VAE Codes
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CodebookMemoryUnit:
    """Codebook index node linking a code ID to its associated atomic facts."""
    code_id: int  # 核心 ID，例如 512
    contained_atomic_ids: List[str] = field(default_factory=list)  # 该码本下挂载的所有事实ID

    # 可选：未来可以利用 LLM 为每个 Code 生成语义描述 (e.g., "Medical Topics")
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 测试代码
if __name__ == '__main__':
    print("=== 测试 Memory Structures (Codebook Layer 版) ===")

    # 1. Base
    base = BaseMemoryUnit(text="I went to the hospital.", source_document="doc1")
    print(f"[Base] {base.id}: {base.text}")

    # 2. Atomic (假设 VQ-VAE 输出代码 [101, 55])
    atomic = AtomicMemoryUnit(
        fact_text="User visited hospital.",
        source_base_ids=[base.id],
        code_indices=[101, 55]
    )
    print(f"[Atomic] {atomic.id}: Codes={atomic.code_indices}")

    # 3. Codebook Layer
    # 系统应该自动归类：Code 101 包含了这个事实
    code_unit_101 = CodebookMemoryUnit(
        code_id=101,
        contained_atomic_ids=[atomic.id],
        description="Locations/Medical"
    )
    print(f"[Codebook Unit #{code_unit_101.code_id}] 包含事实数: {len(code_unit_101.contained_atomic_ids)}")
