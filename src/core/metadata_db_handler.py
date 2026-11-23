# src/core/metadata_db_handler.py

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Any, Dict
from collections import defaultdict

from src.core.memory_structures import BaseMemoryUnit, AtomicMemoryUnit


def _datetime_serializer(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class MetadataDBHandler:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # 统计文件路径：db_path + .stats.json
        self.stats_path = f"{db_path}.stats.json"

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

        # 初始化全局统计数据结构
        # total_docs: atomic-document count, the numerator in the IDF ratio.
        # code_freq: document frequency per code, the denominator in the IDF ratio.
        # co_occurrence: {code_A: {code_B: count}} (用于共现概率 P(B|A))
        self.stats = {
            "total_docs": 0,
            "code_freq": defaultdict(int),
            "co_occurrence": defaultdict(lambda: defaultdict(int))
        }

        # 加载持久化的统计数据
        self._load_stats()

        print(f"元数据数据库已连接: {db_path} (Stats Loaded)")

    def _create_tables(self):
        # 1. 基础记忆
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS base_memory (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            source_document TEXT,
            timestamp TEXT NOT NULL,
            metadata TEXT
        )''')

        # 2. 原子事实 (带 code_indices 存储)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS atomic_memory (
            id TEXT PRIMARY KEY,
            fact_text TEXT NOT NULL,
            code_indices TEXT, -- JSON List
            metadata TEXT
        )''')

        # 3. 关联: Atomic -> Base
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS atomic_to_base_links (
            atomic_id TEXT,
            base_id TEXT,
            PRIMARY KEY (atomic_id, base_id),
            FOREIGN KEY (atomic_id) REFERENCES atomic_memory(id),
            FOREIGN KEY (base_id) REFERENCES base_memory(id)
        )''')

        # 4. 码本记忆层
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS codebook_memory (
            code_id INTEGER PRIMARY KEY,
            description TEXT,
            metadata TEXT
        )''')

        # 5. 码本 -> 事实 关联表 (倒排索引)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS code_to_atomic_links (
            code_id INTEGER,
            atomic_id TEXT,
            PRIMARY KEY (code_id, atomic_id),
            FOREIGN KEY (code_id) REFERENCES codebook_memory(code_id),
            FOREIGN KEY (atomic_id) REFERENCES atomic_memory(id)
        )''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_code_lookup ON code_to_atomic_links (code_id)')

        self.conn.commit()

    # =========================================================================
    # 统计状态机 (Statistics State Machine)
    # =========================================================================

    def _load_stats(self):
        """从 JSON 文件加载统计数据，并处理 Key 类型转换 (Str -> Int)"""
        if not os.path.exists(self.stats_path):
            return

        try:
            with open(self.stats_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            self.stats["total_docs"] = raw_data.get("total_docs", 0)

            # JSON 的 Key 只能是 String，加载后需要转回 Int
            # 1. 恢复 code_freq
            loaded_freq = raw_data.get("code_freq", {})
            for k, v in loaded_freq.items():
                self.stats["code_freq"][int(k)] = v

            # 2. 恢复 co_occurrence
            loaded_co = raw_data.get("co_occurrence", {})
            for k_a, neighbors in loaded_co.items():
                code_a = int(k_a)
                for k_b, count in neighbors.items():
                    self.stats["co_occurrence"][code_a][int(k_b)] = count

        except Exception as e:
            print(f"⚠️ [Stats] 加载统计文件失败: {e}，将使用空统计启动。")

    def _save_stats(self):
        """将内存中的统计数据序列化到 JSON"""
        try:
            # 准备可序列化的数据 (defaultdict 转 dict)
            save_data = {
                "total_docs": self.stats["total_docs"],
                "code_freq": dict(self.stats["code_freq"]),
                "co_occurrence": {k: dict(v) for k, v in self.stats["co_occurrence"].items()}
            }

            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False)

        except Exception as e:
            print(f"❌ [Stats] 保存统计文件失败: {e}")

    def update_stats(self, new_facts: List[AtomicMemoryUnit]):
        """
        [关键] 在线增量更新统计数据
        该方法应在 Pipeline 写入 DB 后立即调用。
        纯内存操作，极快。
        """
        if not new_facts: return

        for amu in new_facts:
            # 1. 更新总文档数
            self.stats["total_docs"] += 1

            # 拿到去重后的 Code 列表
            unique_codes = sorted(list(set(amu.code_indices))) if amu.code_indices else []

            # 2. 更新单频
            for code in unique_codes:
                self.stats["code_freq"][code] += 1

            # 3. 更新共现 (两两组合)
            # 使用双重循环更新无向图 A-B 和 B-A
            for i in range(len(unique_codes)):
                code_a = unique_codes[i]
                for j in range(i + 1, len(unique_codes)):
                    code_b = unique_codes[j]

                    self.stats["co_occurrence"][code_a][code_b] += 1
                    self.stats["co_occurrence"][code_b][code_a] += 1

    def get_stats(self) -> Dict[str, Any]:
        """供 QueryEngine 获取当前统计快照"""
        return self.stats

    # =========================================================================
    # 原有 CRUD 方法
    # =========================================================================

    def add_base_memory(self, bmu: BaseMemoryUnit):
        ts_str = bmu.timestamp.isoformat() if isinstance(bmu.timestamp, datetime) else str(bmu.timestamp)
        self.cursor.execute(
            "INSERT INTO base_memory (id, text, source_document, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
            (bmu.id, bmu.text, bmu.source_document, ts_str,
             json.dumps(bmu.metadata, default=_datetime_serializer))
        )
        self.conn.commit()

    def get_base_memory_by_id(self, bmu_id: str) -> Optional[BaseMemoryUnit]:
        self.cursor.execute("SELECT id, text, source_document, timestamp, metadata FROM base_memory WHERE id=?",
                            (bmu_id,))
        row = self.cursor.fetchone()
        if row: return BaseMemoryUnit(id=row[0], text=row[1], source_document=row[2],
                                      timestamp=datetime.fromisoformat(row[3]), metadata=json.loads(row[4]))
        return None

    def add_atomic_memory(self, amu: AtomicMemoryUnit):
        with self.conn:
            # 1. 存原子事实
            self.cursor.execute(
                "INSERT INTO atomic_memory (id, fact_text, code_indices, metadata) VALUES (?, ?, ?, ?)",
                (amu.id, amu.fact_text, json.dumps(amu.code_indices),
                 json.dumps(amu.metadata, default=_datetime_serializer))
            )

            # 2. 存来源关联
            if amu.source_base_ids:
                links = [(amu.id, bid) for bid in amu.source_base_ids]
                self.cursor.executemany(
                    "INSERT INTO atomic_to_base_links (atomic_id, base_id) VALUES (?, ?)", links
                )

            # 3. 更新码本层 (倒排索引)
            if amu.code_indices:
                unique_codes = set(amu.code_indices)

                # A. 确保 Codebook Unit 存在
                code_units = [(c, None, "{}") for c in unique_codes]
                self.cursor.executemany(
                    "INSERT OR IGNORE INTO codebook_memory (code_id, description, metadata) VALUES (?, ?, ?)",
                    code_units
                )

                # B. 建立 Code -> Atomic 链接
                code_links = [(c, amu.id) for c in unique_codes]
                self.cursor.executemany(
                    "INSERT OR IGNORE INTO code_to_atomic_links (code_id, atomic_id) VALUES (?, ?)",
                    code_links
                )

    def get_atomic_memory_by_id(self, amu_id: str) -> Optional[AtomicMemoryUnit]:
        self.cursor.execute("SELECT id, fact_text, code_indices, metadata FROM atomic_memory WHERE id=?", (amu_id,))
        row = self.cursor.fetchone()
        if not row: return None

        # 获取关联 Base ID
        self.cursor.execute("SELECT base_id FROM atomic_to_base_links WHERE atomic_id=?", (amu_id,))
        base_ids = [r[0] for r in self.cursor.fetchall()]

        codes = json.loads(row[2]) if row[2] else []

        return AtomicMemoryUnit(
            id=row[0], fact_text=row[1], code_indices=codes,
            metadata=json.loads(row[3]), source_base_ids=base_ids
        )

    def get_atomic_ids_by_code(self, code_id: int) -> List[str]:
        self.cursor.execute("SELECT atomic_id FROM code_to_atomic_links WHERE code_id=?", (code_id,))
        return [r[0] for r in self.cursor.fetchall()]

    def get_all_active_atomics(self) -> List[AtomicMemoryUnit]:
        self.cursor.execute("SELECT id, fact_text, code_indices, metadata FROM atomic_memory")
        rows = self.cursor.fetchall()

        self.cursor.execute("SELECT atomic_id, base_id FROM atomic_to_base_links")
        link_rows = self.cursor.fetchall()
        links_map = {}
        for aid, bid in link_rows:
            if aid not in links_map: links_map[aid] = []
            links_map[aid].append(bid)

        results = []
        for row in rows:
            meta_str = row[3]
            meta = json.loads(meta_str) if meta_str else {}
            if meta.get('status') == 'archived': continue

            a_id = row[0]
            codes = json.loads(row[2]) if row[2] else []
            base_ids = links_map.get(a_id, [])

            amu = AtomicMemoryUnit(
                id=a_id,
                fact_text=row[1],
                code_indices=codes,
                metadata=meta,
                source_base_ids=base_ids,
                embedding=None
            )
            results.append(amu)
        return results

    def archive_atomic_memory(self, memory_id: str, reason: str = "Evolved", extra_updates: dict = None):
        with self.conn:
            self.cursor.execute("SELECT metadata FROM atomic_memory WHERE id = ?", (memory_id,))
            row = self.cursor.fetchone()
            if row:
                current_meta = json.loads(row[0]) if row[0] else {}
                updates = {
                    'status': 'archived',
                    'archived_at': datetime.now().isoformat(),
                    'archive_reason': reason
                }
                current_meta.update(updates)
                if extra_updates:
                    current_meta.update(extra_updates)

                self.cursor.execute(
                    "UPDATE atomic_memory SET metadata = ? WHERE id = ?",
                    (json.dumps(current_meta, default=_datetime_serializer), memory_id)
                )

    def get_total_atomic_count(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM atomic_memory")
        result = self.cursor.fetchone()
        return result[0] if result else 0

    def update_atomic_memory_metadata(self, memory_id: str, new_metadata: Dict[str, Any]):
        """Serialize and update fact metadata, including consolidation pointers."""
        with self.conn:
            self.cursor.execute(
                "UPDATE atomic_memory SET metadata = ? WHERE id = ?",
                (json.dumps(new_metadata, default=_datetime_serializer), memory_id)
            )

    def close(self):
        # 关闭时保存统计数据，防止丢失
        self._save_stats()
        if self.conn:
            self.conn.close()
