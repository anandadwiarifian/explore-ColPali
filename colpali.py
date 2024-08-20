from typing import List, Any, Union
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from PIL import Image
from dotenv import load_dotenv

from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from colpali_engine.utils.image_from_page_utils import (
    load_from_pdf, load_from_dataset, load_from_image_urls
)

load_dotenv()

class ColPaliModel:
    def __init__(self) -> None:
        self.model_name = "vidore/colpali"
        self.model = ColPali.from_pretrained(
            "google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16, device_map="cpu").eval()
        self.model.load_adapter(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.ds = []

    def load_dataset(self, file_path: str) -> None:
        files = load_from_dataset(file_path)
        self._load(files)

    def load_image_urls(self, file_path: str) -> None:
        files = load_from_image_urls(file_path)
        self._load(files)     

    def load_pdfs(self, file_path: str) -> None:
        files = load_from_pdf(file_path)
        self._load(files)

    def _load(self, files: Union[List,Any]) -> None:
        dataloader = DataLoader(
            files,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: process_images(self.processor, x),
        )

        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            self.ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))


    def query(self, queries: Union[str, List[str]]) -> List:
        if isinstance(queries, str):
            queries = [queries]

        dataloader = DataLoader(
            queries,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: process_queries(
                self.processor, x, Image.new("RGB", (448, 448), (255, 255, 255))),
        )

        qs = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        # run evaluation
        retriever_evaluator = Evaluator(is_multi_vector=True)
        return retriever_evaluator.evaluate(qs, self.ds, "cpu")

class Evaluator(CustomEvaluator):
    def evaluate_colbert(self, qs, ps, device_map, batch_size=128) -> torch.Tensor:
        scores = []
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device_map
            )
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(device_map)
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores.append(scores_batch)
        scores = torch.cat(scores, dim=0)
        return scores

    def evaluate(self, qs, ps, device_map: str):
        if self.is_multi_vector:
            scores = self.evaluate_colbert(qs, ps, device_map)
        else:
            scores = self.evaluate_biencoder(qs, ps)

        assert scores.shape[0] == len(qs)

        arg_score = scores.argmax(dim=1)
        # compare to arange
        accuracy = (arg_score == torch.arange(scores.shape[0], device=scores.device)).sum().item() / scores.shape[0]
        print(arg_score)
        print(f"Top 1 Accuracy (verif): {accuracy}")

        # cast to numpy
        # scores = scores.cpu().numpy()
        scores = scores.to(torch.float32).cpu().numpy()
        return scores