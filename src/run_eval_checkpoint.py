import torch
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments

# Config
dataset = 'MNIST'
model = 'ViT-L-14'
args = parse_arguments()
args.data_location = '/path/to/data'
args.model = model
args.save = f'checkpoints/{model}'
pretrained_checkpoint = f'src/checkpoints/ViT-B-16-20260113T192236Z-1-001/ViT-B-16/zeroshot.pt'
finetuned_checkpoint = f'src/checkpoints/ViT-B-16-20260113T192236Z-1-001/ViT-B-16/EuroSAT/finetuned.pt'


# Create the task vector
task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint)
neg_task_vector = -task_vector
# Apply the task vector
image_encoder = neg_task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.5)
# Evaluate
eval_single_dataset(image_encoder, dataset, args)
eval_single_dataset(image_encoder, 'ImageNet', args)
