import os
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# 数据库持久化文件夹
CHROMA_DB_PATH = "./chroma_db"
# 集合名称
COLLECTION_NAME = "baduanjin_poses"
BADUANJIN_MAPPING = {
    "action1": "两手托天理三焦",
    "action2": "左右开弓似射雕",
    "action3": "调理脾胃须单举",
    "action4": "五劳七伤往后瞧",
    "action5": "摇头摆尾去心火",
    "action6": "两手攀足固肾腰",
    "action7": "攒拳怒目增气力",
    "action8": "背后七颠百病消",
}

# 初始化Chroma
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# 创建/获取集合（维度要和特征向量一致=132）
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},  # 使用余弦相似度，适合姿势识别
)


def add_pose_to_chroma(vector_path, pose_name):
    """单个特征向量写入chroma"""
    # 读取向量转成lsit
    vector = np.load(vector_path).tolist()
    # 唯一ID
    doc_id = os.path.splitext(os.path.basename(vector_path))[0]
    # 必须传list类型
    collection.add(
        ids=[doc_id],
        documents=[pose_name],
        embeddings=[vector],
        metadatas=[{"pose_name": pose_name, "source_file": vector_path}],
    )


def query_similar_poses(query_vector_path, top_k=5):
    """查询最相似的姿势"""
    query_vector = np.load(query_vector_path).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["metadatas", "distances"],  # 返回元数据和相似度
    )
    print(f"查询结果(Top-{top_k}):")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity = 1 - distance  # 余弦距离转相似度
        print(f"{i + 1}. {meta['pose_name']} | 相似度: {similarity:.4f})")


if __name__ == "__main__":
    # 批量入库
    fetures_dir = "./features"
    for file in tqdm(os.listdir(fetures_dir)):
        if file.endswith(".npy"):
            vector_path = os.path.join(fetures_dir, file)
            pose_name = BADUANJIN_MAPPING[os.path.splitext(file)[0].split("_")[0]]
            add_pose_to_chroma(vector_path, pose_name)

    # # 测试查询
    # test_query = "./features/action1_1.npy"
    # if os.path.exists(test_query):
    #     query_similar_poses(test_query, top_k=3)
    print(len(collection))
