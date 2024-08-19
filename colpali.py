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
            "google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16, device_map="cuda").eval()
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
        retriever_evaluator = CustomEvaluator(is_multi_vector=True)
        return retriever_evaluator.evaluate(qs, self.ds)
