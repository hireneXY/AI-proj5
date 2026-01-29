import torch
import numpy as np
import random

def set_seed(seed=42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Set seed to {seed} for reproducibility")

def seed_worker(worker_id):
    """为DataLoader设置worker种子"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader_with_seed(dataset, batch_size, seed=42):
    """获取带有固定种子的DataLoader"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator,
        num_workers=4,
        pin_memory=True
    )