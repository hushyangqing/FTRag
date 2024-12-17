from transformers import (
    RagRetriever,
    AutoModelForCausalLM,
    AutoTokenizer,
    DPRQuestionEncoderTokenizer,
    DPRQuestionEncoder,
    RagConfig,
    RagTokenForGeneration,
    DPRContextEncoder
)
from datasets import Dataset, load_dataset, load_from_disk
import torch
from typing import Dict, List
import os
import faiss

class RAGSystem:
    def __init__(
        self,
        retrieval_corpus_path: str = "./retrieval_corpus",
        faiss_index_path: str = "./faiss_index",
        fine_tuned_model_path: str = "./fine_tuned_model",
        n_docs: int = 2,
        max_length: int = 300
    ):
        self.retrieval_corpus_path = retrieval_corpus_path
        self.faiss_index_path = faiss_index_path
        self.fine_tuned_model_path = fine_tuned_model_path
        self.n_docs = n_docs
        self.max_length = max_length
        
        self.question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        self.question_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        self.generator_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
        if self.generator_tokenizer.pad_token is None:
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
        self.generator = AutoModelForCausalLM.from_pretrained(
            self.fine_tuned_model_path,
            from_tf=True
        )

    def load_data(self, sample_size: int = 1000) -> Dataset:
        print(f"Loading dataset with sample size: {sample_size}")
        dataset = load_dataset(
            "CShorten/ML-ArXiv-Papers",
            split=f"train[:{sample_size}]"
        )

        # dataset = load_dataset(
        #     "CShorten/ML-ArXiv-Papers",
        #     split="train"
        # )

        retrieval_corpus = Dataset.from_dict({
            "title": dataset["title"],
            "text": dataset["abstract"],
            "id": range(len(dataset))
        })

        retrieval_corpus = self.add_embeddings(retrieval_corpus)

        os.makedirs(self.retrieval_corpus_path, exist_ok=True)
        retrieval_corpus.save_to_disk(self.retrieval_corpus_path)
        return retrieval_corpus
    
    def add_embeddings(self, dataset: Dataset) -> Dataset:
        context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        
        def embed_text(examples: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
            inputs = self.question_encoder_tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                embeddings = context_encoder(**inputs).pooler_output.numpy()
            return {"embeddings": embeddings}
        
        dataset = dataset.map(
            embed_text,
            batched=True,
            batch_size=8
        )
        return dataset

    def build_faiss_index(self) -> None:
        retrieval_corpus = load_from_disk(self.retrieval_corpus_path)
        
        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
        retrieval_corpus.add_faiss_index(column="embeddings")
        retrieval_corpus.get_index("embeddings").save(self.faiss_index_path)

        print(f"Index saved to {self.faiss_index_path}")

    def initialize_rag_model(self) -> RagTokenForGeneration:
        generator = AutoModelForCausalLM.from_pretrained(self.fine_tuned_model_path, from_tf=True)
        
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-base",
            index_name="custom",
            passages_path=self.retrieval_corpus_path,
            index_path=self.faiss_index_path
        )
        retriever.config.n_docs = self.n_docs
        
        config = RagConfig.from_pretrained("facebook/rag-token-nq") 
        
        rag_model = RagTokenForGeneration(
            config=config,
            question_encoder=self.question_encoder,
            generator=generator,
            retriever=retriever
        )
        
        return rag_model
    
    def retrieve(self, query: str) -> List[str]:

        retrieval_corpus = load_from_disk(self.retrieval_corpus_path)
        
        # Load and add the saved FAISS index
        retrieval_corpus.load_faiss_index('embeddings', self.faiss_index_path)

        # Encode query
        query_inputs = self.question_encoder_tokenizer(
            query,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            query_embeddings = self.question_encoder(**query_inputs).pooler_output.numpy()
        
        scores, retrieved_examples = retrieval_corpus.get_nearest_examples(
            'embeddings',
            query_embeddings,
            k=self.n_docs
        )
        
        return retrieved_examples['text']

    def generate(self, query: str, contexts: List[str]) -> str:
        prompt = "Context:\n"
        for i, ctx in enumerate(contexts, 1):
            prompt += f"{i}. {ctx}\n"
        prompt += f"\nQuestion: {query}\nAnswer:"
        
        inputs = self.generator_tokenizer(
            prompt,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.generator.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length + inputs["input_ids"].shape[1],
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.generator_tokenizer.eos_token_id
            )
        
        generated_text = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(self.generator_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        
        return answer.strip()

def run_rag_pipeline(query: str = "What is the purpose of clustering in machine learning?"):
    try:
        rag_system = RAGSystem()
        
        print("Loading and processing corpus...")
        rag_system.load_data()
        
        print("Building FAISS index...")
        rag_system.build_faiss_index()
        
        print("Initializing RAG model...")
        model = rag_system.initialize_rag_model()
        
        print("Retrieving relevant contexts...")
        contexts = rag_system.retrieve(query)

        print("Generating response...")
        response = rag_system.generate(query, contexts)
        
        print(f"\nQuery: {query}")
        print(f"Response: {response}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    run_rag_pipeline()