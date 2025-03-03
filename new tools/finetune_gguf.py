from llama_cpp import Llama
import json
from transformers import Trainer, TrainingArguments

def load_dataset(data_path):
    with open(data_path, 'r') as f:
        return [json.loads(line) for line in f]

def finetune_model(model_path, data_path, epochs=3, lr=2e-5):
    # Initialize model
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1
    )
    
    # Load and preprocess data
    dataset = load_dataset(data_path)
    
    # Define training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        learning_rate=lr,
        logging_dir='./logs'
    )
    
    # Create Trainer instance
    trainer = Trainer(
        model=llm,
        args=training_args,
        train_dataset=dataset
    )
    
    # Start training
    trainer.train()
    
    # Save fine-tuned model
    output_path = f"finetuned_{os.path.basename(model_path)}"
    trainer.save_model(output_path)
    return output_path

def start_finetune(**kwargs):
    try:
        finetune_model(**kwargs)
    except Exception as e:
        print(f"Finetuning failed: {str(e)}")