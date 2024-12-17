import os
import math
import tensorflow as tf
import pandas as pd
from typing import Tuple, Dict
from transformers import (
    TFAutoModelForCausalLM,
    AutoTokenizer,
    AdamWeightDecay,
    pipeline,
    create_optimizer
)
from datasets import Dataset, load_dataset
import plotly.express as px

class ModelTrainer:
    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_length: int = 300,
        batch_size: int = 16,
        save_directory: str = "./fine_tuned_model"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.save_directory = save_directory
        self.tokenizer = None
        self.model = None
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def load_model(self) -> None:
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = TFAutoModelForCausalLM.from_pretrained(
            self.model_name,
            pad_token_id=self.tokenizer.eos_token_id
        )
        print(f"Model '{self.model_name}' and tokenizer loaded successfully.")

    def load_and_process_data(
        self,
        dataset_name: str,
        train_size: int = 1000,
        val_size: int = 200
    ) -> Tuple[Dataset, Dataset]:
        print(f"Loading dataset: {dataset_name}")
        
        data = load_dataset(dataset_name, split='train')
        data = data.train_test_split(shuffle=True, seed=200, test_size=0.2)
        
        train = data["train"].select(range(train_size))
        val = data["test"].select(range(val_size))
        
        abstracts = [len(x.split()) for x in data["train"]["abstract"]]
        length_hist = px.histogram(
            abstracts,
            nbins=400,
            marginal="rug",
            labels={"value": "Article Length (words)"}
        )
        length_hist.show()
        
        return train, val

    def tokenize_data(self, dataset: Dataset) -> Dataset:
        def tokenization(examples):
            return self.tokenizer(
                examples["abstract"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
        
        def create_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        columns_to_remove = ["title", "abstract", "Unnamed: 0", "Unnamed: 0.1"]
        tokenized = dataset.map(
            tokenization,
            batched=True,
            remove_columns=columns_to_remove
        )
        
        return tokenized.map(create_labels, batched=True, num_proc=10)

    def prepare_training(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        train_set = self.model.prepare_tf_dataset(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size
        )
        
        validation_set = self.model.prepare_tf_dataset(
            val_dataset,
            shuffle=False,
            batch_size=self.batch_size
        )
        
        return train_set, validation_set

    def setup_training(self) -> None:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0005,
            decay_steps=500,
            decay_rate=0.95,
            staircase=False
        )
        
        optimizer = AdamWeightDecay(
            learning_rate=lr_schedule,
            weight_decay_rate=0.01
        )
        
        for layer in self.model.layers[:-2]:
            layer.trainable = False
            
        self.model.compile(optimizer=optimizer)

    def train(
        self,
        train_set: tf.data.Dataset,
        validation_set: tf.data.Dataset,
        epochs: int = 1
    ) -> None:
        history = self.model.fit(
            train_set,
            validation_data=validation_set,
            epochs=epochs,
            workers=9,
            use_multiprocessing=True
        )
        
        # Evaluate
        eval_loss = self.model.evaluate(validation_set)
        print(f"Perplexity: {math.exp(eval_loss):.2f}")
        
        # Save model
        self.model.save_pretrained(self.save_directory)
        self.tokenizer.save_pretrained(self.save_directory)
        print(f"Model and tokenizer saved to {self.save_directory}")

    def test_generation(self, test_sentence: str = "clustering") -> None:
        text_generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="tf",
            max_new_tokens=500
        )
        
        generated_output = text_generator(test_sentence)
        for idx, output in enumerate(generated_output):
            print(f"Generated Text {idx + 1}:\n{output['generated_text']}\n")

def main():
    trainer = ModelTrainer()
    
    try:
        # Load model
        trainer.load_model()
        
        # Load and process data
        train_data, val_data = trainer.load_and_process_data("CShorten/ML-ArXiv-Papers")
        
        # Tokenize data
        train_tokenized = trainer.tokenize_data(train_data)
        val_tokenized = trainer.tokenize_data(val_data)
        
        # Prepare for training
        train_set, val_set = trainer.prepare_training(train_tokenized, val_tokenized)
        
        # Setup and train
        trainer.setup_training()
        trainer.train(train_set, val_set)
        
        # Test
        trainer.test_generation()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()