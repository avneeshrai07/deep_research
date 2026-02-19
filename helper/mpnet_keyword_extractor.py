from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Dict
from sklearn.cluster import DBSCAN
import numpy as np
from helper.mpnet_helper import mpnet_helper_dict


# -------------------- Main Extractor --------------------


class MPNetExtractor:
    def __init__(
        self,
        user_intent: str = "user_post",
        model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        high_priority_weight: float = 2.0,
        device: str = None
    ):
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load SentenceTransformer model '{model_name}': {e}")

        self.high_weight = high_priority_weight

        try:
            intent_config = mpnet_helper_dict[user_intent]
            self.high_priority = intent_config["high_priority_keywords"]
            self.exclude_keywords = intent_config["exclude_keywords"]
            print(f"✅ Loaded intent '{user_intent}': {len(self.high_priority)} high_priority, {len(self.exclude_keywords)} exclude keywords")
        except KeyError as e:
            raise KeyError(f"❌ Missing key in mpnet_helper_dict for intent '{user_intent}': {e}")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load intent config for '{user_intent}': {e}")

        try:
            self.high_priority_emb = self.model.encode(
                self.high_priority,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ) if self.high_priority else None
            print(f"✅ high_priority_emb shape: {self.high_priority_emb.shape if self.high_priority_emb is not None else 'None'}")
        except Exception as e:
            print(f"❌ Failed to encode high_priority keywords: {e}")
            self.high_priority_emb = None

        try:
            self.exclude_emb = self.model.encode(
                self.exclude_keywords,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ) if self.exclude_keywords else None
            print(f"✅ exclude_emb shape: {self.exclude_emb.shape if self.exclude_emb is not None else 'None'}")
        except Exception as e:
            print(f"❌ Failed to encode exclude keywords: {e}")
            self.exclude_emb = None


    def _encode_documents(self, documents: List[Dict], batch_size: int = 32) -> torch.Tensor:
        """Encode documents — uses 'title' + 'text' if available, else any string fields joined."""
        try:
            texts = []
            for doc in documents:
                try:
                    if "title" in doc and "text" in doc:
                        texts.append(f"{doc['title']}. {doc['text']}")
                    elif "text" in doc:
                        texts.append(doc["text"])
                    elif "title" in doc:
                        texts.append(doc["title"])
                    else:
                        texts.append(" ".join(str(v) for v in doc.values() if isinstance(v, str)))
                except Exception as e:
                    print(f"❌ Failed to extract text from document '{doc}': {e}")
                    texts.append("")  # preserve index alignment

            return self.model.encode(
                texts,
                convert_to_tensor=True,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(documents) > 100
            )
        except Exception as e:
            raise RuntimeError(f"❌ _encode_documents failed: {e}")


    def _filter_excluded(
        self,
        documents: List[Dict],
        doc_embeddings: torch.Tensor,
        exclusion_threshold: float = 0.6
    ) -> tuple[List[Dict], torch.Tensor]:
        """Remove documents that match exclusion keywords."""
        try:
            if self.exclude_emb is None:
                return documents, doc_embeddings

            keep_indices = []
            for i, doc in enumerate(documents):
                try:
                    exclude_sims = util.cos_sim(doc_embeddings[i], self.exclude_emb)[0]
                    if torch.max(exclude_sims).item() < exclusion_threshold:
                        keep_indices.append(i)
                except Exception as e:
                    print(f"❌ Failed to compute exclusion similarity for doc index {i}: {e}")
                    keep_indices.append(i)  # keep on error to avoid silent data loss

            filtered_docs = [documents[i] for i in keep_indices]
            filtered_emb = doc_embeddings[keep_indices]

            removed = len(documents) - len(filtered_docs)
            if removed > 0:
                print(f"🧹 Excluded: {len(documents)} → {len(filtered_docs)} documents (removed {removed})")

            return filtered_docs, filtered_emb

        except Exception as e:
            print(f"❌ _filter_excluded failed, returning original documents: {e}")
            return documents, doc_embeddings


    def _score_and_rank(
        self,
        documents: List[Dict],
        doc_embeddings: torch.Tensor,
        top_n: int,
        min_score: float
    ) -> List[Dict]:
        """Score documents against high_priority embeddings and return top N."""
        try:
            scored = []
            for i, doc in enumerate(documents):
                try:
                    high_sims = util.cos_sim(doc_embeddings[i], self.high_priority_emb)[0]
                    score = torch.max(high_sims).item() * self.high_weight
                    print(f"📊 Doc {i} score: {score:.4f} | title: {doc.get('title', '')[:60]}")
                    if score >= min_score:
                        scored.append((score, doc))
                except Exception as e:
                    print(f"❌ Scoring failed for doc index {i}: {e}")
                    continue

            print(f"📊 Scoring complete: {len(scored)} docs passed min_score={min_score} out of {len(documents)}")
            scored.sort(key=lambda x: x[0], reverse=True)

            # Fallback: if nothing passes min_score, return top N by raw score anyway
            if not scored:
                print(f"⚠️ No docs passed min_score={min_score} — returning top N by raw score.")
                all_scored = []
                for i, doc in enumerate(documents):
                    try:
                        high_sims = util.cos_sim(doc_embeddings[i], self.high_priority_emb)[0]
                        score = torch.max(high_sims).item() * self.high_weight
                        all_scored.append((score, doc))
                    except Exception:
                        continue
                all_scored.sort(key=lambda x: x[0], reverse=True)
                return [doc for _, doc in all_scored[:top_n]]

            return [doc for _, doc in scored[:top_n]]

        except Exception as e:
            print(f"❌ _score_and_rank failed: {e}")
            return documents[:top_n]


    def extract(
        self,
        documents: List[Dict],
        top_n: int = 10,
        min_score: float = 0.3,
        exclusion_threshold: float = 0.6,
        batch_size: int = 32,
        cluster: bool = False,
        cluster_eps: float = 0.3,
        cluster_min_samples: int = 3
    ) -> List[Dict]:
        try:
            if not documents:
                return []

            # ── MODE 3: Both empty → return top N as-is ──
            if self.high_priority_emb is None and self.exclude_emb is None:
                print("ℹ️ No keywords provided — returning top N documents as-is.")
                return documents[:top_n]

            try:
                doc_embeddings = self._encode_documents(documents, batch_size)
            except Exception as e:
                print(f"❌ extract() failed at document encoding: {e}")
                return documents[:top_n]  # fallback to top N as-is

            # ── MODE 1: Only exclude keywords → filter out excluded, return top N ──
            if self.high_priority_emb is None:
                print("ℹ️ No high_priority keywords — filtering excluded docs and returning top N.")
                try:
                    documents, _ = self._filter_excluded(
                        documents, doc_embeddings, exclusion_threshold
                    )
                except Exception as e:
                    print(f"❌ extract() failed at exclusion filtering (mode 1): {e}")
                return documents[:top_n]

            # ── MODE 2: high_priority given → apply exclusion if present, then score ──
            if self.exclude_emb is not None:
                try:
                    documents, doc_embeddings = self._filter_excluded(
                        documents, doc_embeddings, exclusion_threshold
                    )
                except Exception as e:
                    print(f"❌ extract() failed at exclusion filtering (mode 2): {e}")

            if not documents:
                return []

            # ── Cluster mode ──
            if cluster:
                try:
                    if len(documents) < cluster_min_samples:
                        print(f"⚠️ Not enough documents ({len(documents)}) for clustering (min={cluster_min_samples}) — falling back to score mode.")
                        return self._score_and_rank(documents, doc_embeddings, top_n, min_score)

                    embeddings_np = doc_embeddings.cpu().numpy()
                    clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples, metric='cosine')
                    labels = clustering.fit_predict(embeddings_np)

                    unique_labels = labels[labels != -1]
                    if len(unique_labels) == 0:
                        print("⚠️ DBSCAN found no clusters (all points are noise) — falling back to score mode.")
                        return self._score_and_rank(documents, doc_embeddings, top_n, min_score)

                    unique_clusters, counts = np.unique(unique_labels, return_counts=True)
                    largest_label = unique_clusters[np.argmax(counts)]
                    cluster_indices = np.where(labels == largest_label)[0]

                    num_to_return = min(len(cluster_indices), top_n)
                    return [documents[i] for i in cluster_indices[:num_to_return]]

                except Exception as e:
                    print(f"❌ extract() failed during clustering — falling back to score mode: {e}")
                    return self._score_and_rank(documents, doc_embeddings, top_n, min_score)

            # ── Score mode (default) ──
            return self._score_and_rank(documents, doc_embeddings, top_n, min_score)

        except Exception as e:
            print(f"❌ extract() unexpected failure: {e}")
            return []


    def extract_top_n(
        self,
        documents: List[Dict],
        top_n: int = 3,
        min_score: float = 0.3,
        cluster: bool = False,
        use_exclusion: bool = True,
        batch_size: int = 32
    ) -> List[Dict]:
        try:
            return self.extract(
                documents=documents,
                top_n=top_n,
                min_score=min_score,
                batch_size=batch_size,
                cluster=False,
            )
        except Exception as e:
            print(f"❌ extract_top_n failed: {e}")
            return []


    def extract_top_cluster(
        self,
        documents: List[Dict],
        top_n: int = 5,
        use_exclusion: bool = True,
        batch_size: int = 32,
        cluster_eps: float = 0.3,
        cluster_min_samples: int = 3
    ) -> List[Dict]:
        try:
            return self.extract(
                documents=documents,
                top_n=top_n,
                batch_size=batch_size,
                cluster=True,
                cluster_eps=cluster_eps,
                cluster_min_samples=cluster_min_samples,
            )
        except Exception as e:
            print(f"❌ extract_top_cluster failed: {e}")
            return []
