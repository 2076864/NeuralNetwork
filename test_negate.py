import torch
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments

# Config
dataset = 'MNIST'
model = 'ViT-B-16'
args = parse_arguments()
args.batch_size = 32
args.data_location = 'Desktop/neuraln/task_vectors/data$'  
args.model = model
args.save = f'checkpoints/{model}'

pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
finetuned_checkpoint = f'checkpoints/{model}/{dataset}/finetuned.pt'

# Test 1: Valuta il modello zero-shot (baseline)
print("\n=== Test 1: Zero-shot baseline ===")
image_encoder = torch.load(pretrained_checkpoint)
eval_single_dataset(image_encoder, dataset, args)

# Test 2: Valuta il modello fine-tuned
print("\n=== Test 2: Fine-tuned model ===")
image_encoder = torch.load(finetuned_checkpoint)
eval_single_dataset(image_encoder, dataset, args)

# Test 3: Nega il task vector (dovrebbe degradare performance)
print("\n=== Test 3: Negated task vector (scaling=-0.5) ===")
task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint)
neg_task_vector = -task_vector
image_encoder = neg_task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.5)
eval_single_dataset(image_encoder, dataset, args)
