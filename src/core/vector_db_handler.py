# src/core/vector_db_handler.py

import os
import json
from typing import List, Tuple, Dict, Any

# 确保已安装 faiss-cpu 和 numpy
# pip install faiss-cpu numpy

try:
    import faiss
    import numpy as np
except ImportError:
    print("错误: 'faiss-cpu' 或 'numpy' 库未安装。")
    print("请运行: pip install faiss-cpu numpy")
    faiss = None
    np = None


class VectorDBHandler:
    """
    负责处理所有与Faiss向量索引的交互。
    管理多个独立的索引，并处理它们的创建、加载、保存和搜索。
    """

    def __init__(self, storage_path: str):
        """
        初始化向量数据库处理器。
        Args:
            storage_path (str): 用于存放所有索引文件的目录。
        """
        if faiss is None or np is None:
            raise ImportError("依赖库 'faiss-cpu' 或 'numpy'未能加载。")

        # 强制转换为绝对路径，防止混合斜杠问题
        self.storage_path = os.path.abspath(storage_path)

        # 再次确保目录存在
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

        self.indexes: Dict[str, Dict[str, Any]] = {}
        print(f"向量数据库处理器已初始化，存储路径: '{self.storage_path}'")

    def _get_paths(self, index_name: str) -> Tuple[str, str]:
        """[内部方法] 获取索引文件和ID映射文件的完整路径。"""
        # 使用 os.path.join 拼接，并用 normpath 标准化分隔符
        index_file = os.path.normpath(os.path.join(self.storage_path, f"{index_name}.faiss"))
        map_file = os.path.normpath(os.path.join(self.storage_path, f"{index_name}_map.json"))
        return index_file, map_file

    def create_or_load_index(self, index_name: str, dimension: int):
        """
        加载一个已存在的索引，或者创建一个新的索引。
        """
        if index_name in self.indexes:
            print(f"索引 '{index_name}' 已加载。")
            return

        index_file, map_file = self._get_paths(index_name)

        if os.path.exists(index_file) and os.path.exists(map_file):
            print(f"正在从 '{index_file}' 加载索引 '{index_name}'...")
            try:
                index = faiss.read_index(index_file)
                with open(map_file, 'r', encoding='utf-8') as f:
                    id_map = {int(k): v for k, v in json.load(f).items()}
                print("加载成功。")
            except Exception as e:
                raise RuntimeError(f"Cannot load existing FAISS index: {index_file}") from e
            if index.d != dimension or index.ntotal != len(id_map):
                raise ValueError("Existing FAISS dimension or ID-map count does not match the memory store")
        else:
            if os.path.exists(index_file) or os.path.exists(map_file):
                raise FileNotFoundError("FAISS index and ID map must exist together")
            print(f"未找到索引 '{index_name}'，正在创建一个新的...")
            index = faiss.IndexFlatL2(dimension)
            id_map = {}
            print(f"创建成功 (维度: {dimension})。")

        self.indexes[index_name] = {
            "index": index,
            "id_map": id_map,
            "modified": False
        }

    def add_vectors(self, index_name: str, vectors: List[Tuple[str, List[float]]]):
        """向指定索引中添加一批向量。"""
        if index_name not in self.indexes:
            raise KeyError(f"索引 '{index_name}' 未被创建或加载。")

        if not vectors: return

        index_info = self.indexes[index_name]
        index = index_info["index"]
        id_map = index_info["id_map"]

        ids_to_add = [v[0] for v in vectors]
        vectors_to_add = np.array([v[1] for v in vectors], dtype='float32')

        if vectors_to_add.shape[1] != index.d:
            raise ValueError(f"向量维度错误！索引 '{index_name}' 需要维度 {index.d}，但提供了 {vectors_to_add.shape[1]}。")

        start_index = index.ntotal
        for i, str_id in enumerate(ids_to_add):
            id_map[start_index + i] = str_id

        index.add(vectors_to_add)
        index_info["modified"] = True
        print(f"已向索引 '{index_name}' 添加 {len(vectors)} 个向量。当前总数: {index.ntotal}。")

    def add_vectors_batch(self, index_name: str, vectors: List[Tuple[str, List[float]]]):
        """批量添加向量"""
        self.add_vectors(index_name, vectors)

    def search(self, index_name: str, query_vector: List[float], k: int) -> List[Tuple[str, float]]:
        """在指定索引中进行相似度搜索。"""
        if index_name not in self.indexes:
            # Unknown index names are handled below.

            raise KeyError(f"索引 '{index_name}' 未被创建或加载。")

        index_info = self.indexes[index_name]
        index = index_info["index"]
        id_map = index_info["id_map"]

        if index.ntotal == 0:
            return []

        query_np = np.array([query_vector], dtype='float32')
        query_np = query_np.reshape(1, -1)

        distances, int_ids = index.search(query_np, k)

        results = []
        for i, int_id in enumerate(int_ids[0]):
            if int_id != -1:
                str_id = id_map.get(int_id)
                if str_id:
                    distance = distances[0][i]
                    results.append((str_id, float(distance)))

        return results

    def save_index(self, index_name: str):
        """将指定的索引和ID映射保存到磁盘。"""
        if index_name not in self.indexes:
            raise KeyError(f"索引 '{index_name}' 不存在。")

        index_info = self.indexes[index_name]
        # 获取标准化后的绝对路径
        index_file, map_file = self._get_paths(index_name)

        print(f"正在将索引 '{index_name}' 保存到 '{index_file}'...")

        # 确保存储目录存在 (双重保险)
        parent_dir = os.path.dirname(index_file)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        try:
            # FAISS C++ 调用
            faiss.write_index(index_info["index"], index_file)

            with open(map_file, 'w', encoding='utf-8') as f:
                json.dump(index_info["id_map"], f)

            index_info["modified"] = False
            print("保存成功。")
        except Exception as e:
            print(f"❌ FAISS 保存失败: {e}")

            raise e


    def get_vectors_by_ids(self, index_name: str, ids: List[str]) -> Dict[str, List[float]]:
        """Retrieve stored FAISS vectors by string fact IDs for consolidation hydration."""
        if index_name not in self.indexes:
            print(f"Warning: Index '{index_name}' not found during retrieval.")
            return {}

        index_info = self.indexes[index_name]
        index = index_info["index"]
        id_map = index_info["id_map"]  # 格式: {int_id: str_id}

        # 1. 构建反向映射 (str_id -> int_id)
        # 注意: 如果数据量极大(百万级)，频繁构建反向表会有性能开销。
        # The reverse map supports string-ID lookup for consolidation.
        reverse_id_map = {v: k for k, v in id_map.items()}

        result = {}

        # 2. 遍历请求的 ID
        for str_id in ids:
            if str_id in reverse_id_map:
                int_id = reverse_id_map[str_id]
                try:
                    # FAISS 的 reconstruct 方法可以从 Flat 索引中直接提取向量
                    vec_np = index.reconstruct(int_id)
                    # 转换为 Python List
                    result[str_id] = vec_np.tolist()
                except Exception as e:
                    print(f"Error reconstructing vector for ID {str_id} (Int: {int_id}): {e}")
            else:
                # 这种情况可能发生：元数据还在，但向量库没存/被删了
                pass

        return result

    def close(self):
        """安全关闭，保存所有被修改过的索引。"""
        print("正在关闭向量数据库处理器...")
        for name in self.indexes:
            if self.indexes[name]["modified"]:
                try:
                    self.save_index(name)
                except Exception as e:
                    print(f"无法保存索引 {name}: {e}")
        print("关闭完成。")
