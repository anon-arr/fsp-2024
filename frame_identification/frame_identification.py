from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from Hydra.models.gold_target_frame_id_token import RobertaForFrameIdentificationWithTargetToken
from Hydra.utils.frame_identification.data import load_dataset, FrameIdentificationDataset
from Hydra.frame_identification.arguments import args
from Hydra.utils.data import get_framenet_metadata
from Hydra.utils.train import compute_metrics
from scipy.special import softmax
from datetime import datetime
import Hydra.config as config
import pandas as pd
import os

os.environ['WANDB_API_KEY'] = 'cc688956ba1b097a3da6f86d4e5c3bf7c86070f8'
os.environ['WANDB_PROJECT'] = 'hydra'
os.environ['WANDB_LOG_MODEL'] = 'true'

# Get model if starting from checkpoint, otherwise use roberta-base
model_name = f'{config.root_path}/{args.model}' if args.model else 'roberta-base'

# Get FrameNet metadata
print(f'Loading FrameNet metadata...')
lu_manager, frame_info, frame_lu_defs = get_framenet_metadata()

# Load dataset
print(f'Loading train data...')
train_data = load_dataset('train_fulltext_data_tokenized.json', args, 
                          lu_manager, frame_info, frame_lu_defs)

print(f'Loading dev data...')
dev_data = load_dataset('dev_fulltext_data_tokenized.json', args,
                        lu_manager, frame_info, frame_lu_defs)

print(f'Loading test data...')
test_data = load_dataset('test_fulltext_data_tokenized.json', args,
                         lu_manager, frame_info, frame_lu_defs)
test_data = test_data[test_data.target_token_span.apply(lambda x: None not in x)].reset_index(drop=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
tokenizer.add_special_tokens({'additional_special_tokens': ['<f>', '</f>', 
                                                            '<lu>', '</lu>', 
                                                            '<e>', '</e>'
                                                            # '<fe>', '</fe>' # Saved for future use
                                                            ]})

# if args.add_info_tokens:
#     tokenizer.add_special_tokens({'additional_special_tokens': 
#                                   [f'<special{i}>' for i in range(args.num_info_tokens)]})

# Create datasets
print(f'Processing datasets...')
train_dataset = FrameIdentificationDataset(train_data, tokenizer, args)
dev_dataset = FrameIdentificationDataset(dev_data, tokenizer, args)
test_dataset = FrameIdentificationDataset(test_data, tokenizer, args)

# Load model
print(f'Loading model...')
if args.target_classifier:
    model = RobertaForFrameIdentificationWithTargetToken.from_pretrained(model_name, num_labels=2)
else:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

model.resize_token_embeddings(len(tokenizer))

# Unique run name
if args.run_name:
    run_name = f'{args.run_name}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
else:
    run_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

# Save run config
with open(f'{config.model_dir}/frame_identification/{run_name}_args.txt', 'w') as f:
    f.write(str(args))

# Set up training arguments
training_args = TrainingArguments(
    report_to = 'wandb',
    run_name=run_name,
    output_dir= f'{config.model_dir}/frame_identification/{run_name}',
    num_train_epochs=args.epochs,
    per_device_train_batch_size=36, # Fits on v100 and T4
    per_device_eval_batch_size=36,
    warmup_ratio=0.05,
    weight_decay=0.001,
    logging_dir=config.logs_dir,
    logging_steps=20,
    evaluation_strategy='steps',
    eval_steps=0.1,
    save_steps=0.1,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    save_strategy='steps',
    lr_scheduler_type='cosine',
    learning_rate=args.lr,
    seed=0,
    data_seed=0
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    compute_metrics=compute_metrics,     # metrics to compute
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset             # evaluation dataset
)

# Train model
if args.train:
    train_result = trainer.train()

    # Log training metrics to file
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    print('Evaluating model on dev set...')
    # Evaluate model
    trainer.evaluate()

if args.test:
    print('Evaluating model on test set...')
    # Predict on test set
    predictions = trainer.predict(test_dataset)
    print(predictions.metrics)

    # Softmax predictions
    preds = softmax(predictions.predictions, axis=-1)

    # Save prediction scores with inputs and labels
    predictions_df = pd.DataFrame({'sentence': test_dataset.sentence,
                                'target_token_span': test_dataset.target_token_span.tolist(),
                                'possible_frame': test_dataset.frame,
                                'frame_definition': test_dataset.frame_definition,
                                'lu': test_dataset.lu,
                                'lu_definition': test_dataset.lu_definition,
                                'label': test_dataset.label,
                                'preds': preds.tolist()})

    predictions_df.to_json(f'{config.model_dir}/frame_identification/{run_name}_predictions.json')
