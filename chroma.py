import chromadb
import argparse
import os

from Voxel_ML import index

from numpy import dot
from numpy.linalg import norm


chroma_client = chromadb.PersistentClient(path="chroma/test/")


def get_client():
    return chroma_client


def get_collection():
    return chroma_client.get_or_create_collection(
        name="test-encodings",
        configuration={
            "hnsw": {
                "space": "cosine",  # Cohere models often use cosine space
                "max_neighbors": 16,
                "num_threads": 4,
            }
        },
    )


def encode_test_data(data_directory):
    collection = get_collection()

    # 1) gather all .stl/.jt/.off under args.data recursively
    files = []
    for root, _, file_names in os.walk(data_directory):
        if "train" in root.lower():
            continue
        for f in file_names:
            if f.lower().endswith((".stl", ".off", ".obj")):
                files.append(os.path.join(root, f))
    if not files:
        print("No meshes found under data")
        return

    embeddings = index(files)

    collection.add(embeddings=embeddings, ids=files)


def query_against_db(input):
    collection = get_collection()

    target = index([input])
    results = collection.query(
        query_embeddings=target, n_results=10, include=["distances"]
    )
    result_ids = results["ids"][0]
    distances = results["distances"][0]
    print("Top 10 matches in descending order: ")
    for i in range(10):
        print(f"id: {result_ids[i]} distance: {distances[i]:.4f}")


def reset_db():
    chroma_client.delete_collection(name="test-encodings")


def compare(input1, input2):

    a, b = index([input1, input2])
    print(
        f"Cosine Similarity between {input1} and {input2} is: {dot(a, b/(norm(a)*norm(b)))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="mode", required=True)

    p_query = subs.add_parser("query")
    p_query.add_argument("-i", "--input", required=True, help="mesh file to encode")

    p_reset = subs.add_parser("reset_db")

    p_encode = subs.add_parser("encode")
    p_encode.add_argument(
        "-d",
        "--data",
        required=True,
        help="path to directory containing files to index",
    )

    p_compare = subs.add_parser("compare")
    p_compare.add_argument("-i1", "--input1", required=True, help="mesh file to encode")
    p_compare.add_argument("-i2", "--input2", required=True, help="mesh file to encode")

    args = parser.parse_args()
    if args.mode == "encode":
        encode_test_data(args.data)
    if args.mode == "reset_db":
        reset_db()
    elif args.mode == "query":
        query_against_db(args.input)
    elif args.mode == "compare":
        compare(args.input1, args.input2)
    else:
        pass
