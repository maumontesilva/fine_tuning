import tensorflow as tf
from datasets import load_dataset
#import pandas as pd

from transformers import GPT2Tokenizer
# using TFGPT2ForSequenceClassification which suports TensorFlow
from transformers import TFGPT2ForSequenceClassification

import evaluate
import numpy as np
import os

# The train part failed due to not enough memory (> 96 Mb)
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Loading the dataset to train our model
dataset = load_dataset("mteb/tweet_sentiment_extraction")
#df = pd.DataFrame(dataset['train'])

#print(f"{df}")

# Load model and tokenizer
model = TFGPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenizer_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenizer_datasets = dataset.map(tokenizer_function, batched=True)

#print(f"{tokenizer_datasets.column_names}")
## {'train': ['id', 'text', 'label', 'label_text', 'input_ids', 'attention_mask'], 'test': ['id', 'text', 'label', 'label_text', 'input_ids', 'attention_mask']}

# Convert dataset to TensorFlow format
def to_tf_dataset(dataset, batch_size):
    # print(f"to_tf_dataset: {dataset.column_names}")
    # ['id', 'label', 'label_text', 'input_ids', 'attention_mask', 'text']
    features = {
        "input_ids": dataset["input_ids"],
        "attention_mask": dataset["attention_mask"]
    }
    labels = dataset["label"]

    """
    dataset=dataset.remove_columns(["id"])
    dataset=dataset.remove_columns(["label"])
    dataset=dataset.remove_columns(["label_text"])
    dataset=dataset.remove_columns(["text"])

    print(f"to_tf_dataset: {dataset.column_names}")
    print(f"INPUT_IDS: {len(dataset["input_ids"])}")       # Should be a list of lists
    print(f"ATTENTION MASK: {len(dataset["attention_mask"])}")       # Should be a list of lists
    print(f"LABEL: {len(label)}")       # Should be a list of lists
    """

    return (
            tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(len(labels))
            .batch(batch_size)
        )

train_dataset = to_tf_dataset(tokenizer_datasets["train"], batch_size=8)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics=["accuracy"]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train model
model.fit(train_dataset, epochs=3)

"""
small_train_dataset = tokenizer_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenizer_datasets["test"].shuffle(seed=42).select(range(1000))

model = TFGPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)

training_args = TFTrainingArguments(
    output_dir="test_trainer",
    #evaluation_strategy="epoch",
    per_device_train_batch_size=1, # Reduce batch size here
    per_device_eval_batch_size=1, #Optionally, reduce for evaluation as well
    gradient_accumulation_steps=4
    )

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    )

trainer.train()
"""
