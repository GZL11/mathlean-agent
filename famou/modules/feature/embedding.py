"""Embedding-based feature extraction module for Famou 2.0."""

from typing import TYPE_CHECKING, List, Optional

from famou.core.data import Context, RolloutResult
from famou.core.protocol import RequiresEmbedding
from famou.modules.feature.base import FeatureModule
from famou.utils.color import Colors

if TYPE_CHECKING:
    from famou.infrastructure.embedding.base import EmbeddingClient


class EmbeddingFeature(FeatureModule, RequiresEmbedding):
    """
    Extract features using embedding models.

    Uses an embedding client (OpenAI, local model, etc.) to generate
    semantic embeddings of the program code.
    """

    embedding_client: "EmbeddingClient" = None # Injected by RolloutEngine

    def extract_features(self, context: Context, result: RolloutResult) -> List[float]:
        """Extract embedding from program code/prompt."""
        # print("Is in embedding.py!!!")
        # print("context:", context)
        program = result.generated_program
        text = program.llm_feedback["key_features"] if program.llm_feedback else program.code

        if not text:
            self.log_warning("Empty text for embedding, using empty code")
            text = "# empty"

        # Generate embedding
        # print("cp1")
        # self.log_info("generating embedding")
        # print("cp2")
        embedding_result = None
        if hasattr(self, 'embedding_client') and self.embedding_client is not None:
            response = self.embedding_client.embed(text)
            embedding_result = response.embedding
        else:
            # response = text
            # print("Th content of text", text)
            print(f"{Colors.RED}[embedding.py] There is no embedding_client, using zero vector{Colors.RESET}")
            # embedding_result = [float(x) for x in text.split()]
            dimension = 1536
            embedding_result = [0.0] * dimension
        # print("response:", response)
        return embedding_result
