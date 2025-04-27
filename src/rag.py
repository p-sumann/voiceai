import os
import pickle
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class RAG:
    def __init__(
        self,
        data_path: str = "data/Agent's manual.txt",
        faiss_dir: str = "src/faiss_index",
        bm25_path: str = "src/bm25.pkl",
        chunk_size: int = 600,
        chunk_overlap: int = 200,
    ):
        # file paths
        self.data_path = data_path
        self.faiss_dir = faiss_dir
        self.bm25_path = bm25_path

        # chunk settings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # placeholders
        self.docs = []
        self.bm25 = None
        self.db = None
        self.embeddings = OpenAIEmbeddings()

        # build or load indices
        self._prepare_documents()
        self._load_or_create_faiss()
        self._load_or_create_bm25()

    def _prepare_documents(self):
        loader = TextLoader(self.data_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.docs = splitter.split_documents(documents)

    def _load_or_create_faiss(self):
        if os.path.isdir(self.faiss_dir):
            # load existing FAISS index
            self.db = FAISS.load_local(
                self.faiss_dir, self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            # create new index
            self.db = FAISS.from_documents(self.docs, self.embeddings)
            os.makedirs(self.faiss_dir, exist_ok=True)
            self.db.save_local(self.faiss_dir)

    def _load_or_create_bm25(self):
        if os.path.isfile(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            # prepare BM25 corpus: split by whitespace
            tokenized = [doc.page_content.split() for doc in self.docs]
            self.bm25 = BM25Okapi(tokenized)
            with open(self.bm25_path, "wb") as f:
                pickle.dump(self.bm25, f)

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        """
        Retrieve top-k docs by FAISS similarity and BM25 ranking, merge results.
        """
        # FAISS retrieval
        faiss_docs = self.db.similarity_search(query, k=k)
        faiss_texts = [d.page_content for d in faiss_docs]

        # BM25 retrieval
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_idxs = bm25_scores.argsort()[-k:][::-1]
        bm25_texts = [self.docs[i].page_content for i in top_idxs]

        # merge, dedupe, and return
        seen = set()
        combined = []
        for text in faiss_texts + bm25_texts:
            if text not in seen:
                seen.add(text)
                combined.append(text)
        return combined[:k]

    def answer(
        self,
        query: str,
        client,
        system_prompt: str,
        k: int = 4,
        max_tokens: int = 300,
    ) -> str:
        # retrieval
        contexts = self.retrieve(query, k=k)
        context_block = "\n\n---\n\n".join(contexts)

        # prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": f"You are given the following context documents to help answer the user query:\n{context_block}",
            },
            {"role": "user", "content": query},
        ]

        # call LLM
        response = (
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
            )
            .choices[0]
            .message.content
        )
        return response
