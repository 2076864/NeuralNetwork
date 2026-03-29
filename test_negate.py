import torch
import gc
import os
import copy
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments

# ========================================================
# FASI ATTIVE
# ========================================================
RUN_BASELINE             = False   # Fase 0: zero-shot puro (α=0)
RUN_ONLY_GTSRB           = False   # Fase 1: solo τ_GTSRB × α
RUN_ONLY_SVHN            = False   # Fase 2: solo τ_SVHN × α
RUN_MULTI                = False   # Fase 3: (τ_GTSRB + τ_SVHN) × α
RUN_WEIGHTED             = False   # Fase 4: τ_GTSRB×α1 + τ_SVHN×α2 (alpha indipendenti)
RUN_NEGATION_GTSRB       = False   # Fase 5: τ_SVHN + (−τ_GTSRB×α)
RUN_NEGATION_SVHN        = False   # Fase 6: τ_GTSRB + (−τ_SVHN×α)
RUN_NEGATION_BASE        = False   # Fase 7: negazione pura sul BASE model ← traccia
RUN_NEGATION_MULTITASK   = False   # Fase 8: negazione selettiva sul MULTI-TASK model ← traccia
# ========================================================

# ========== CONFIG ==========
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
gc.collect()

model = 'ViT-B-16'
args = parse_arguments()
args.batch_size = 128
args.data_location = 'Desktop/neuraln/task_vectors/data$'
args.model = model
args.save = f'checkpoints/{model}'
pretrained = f'checkpoints/{model}/zeroshot.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Alpha per Fasi 1–3 (sweep ampio)
alphas = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

# Alpha indipendenti per Fase 4 — raffinati intorno al sweet spot α=0.5
WEIGHTED_ALPHAS = [
    (0.3, 0.3),
    (0.5, 0.3),
    (0.3, 0.5),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.4, 0.6),
    (0.7, 0.5),
    (0.5, 0.7),
    (0.8, 0.5),
    (0.5, 0.8),
]

# Alpha per negazione (Fasi 5–8)
NEGATION_ALPHAS = [0.5, 1.0, 1.5]

# ========== UTILITY ==========
def get_acc(raw):
    return raw['top1'] if isinstance(raw, dict) else raw

def get_mem():
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

def eval_model(model_obj):
    torch.cuda.empty_cache()
    acc_g = get_acc(eval_single_dataset(model_obj, 'GTSRB', args))
    acc_s = get_acc(eval_single_dataset(model_obj, 'SVHN', args))
    mem   = get_mem()
    return acc_g, acc_s, mem

def print_header(title, note=""):
    print(f"\n{'='*60}")
    print(f"  {title}")
    if note:
        print(f"  ℹ {note}")
    print(f"{'='*60}")
    print(f"  {'label':<16} {'GTSRB':>8} {'SVHN':>8} {'GPU Mem':>10}")
    print(f"  {'-'*46}")

def print_row(label, acc_g, acc_s, mem):
    print(f"  {label:<16} {acc_g*100:>7.1f}% {acc_s*100:>7.1f}% {mem:>8.2f}GB")

def cleanup(model_obj):
    del model_obj
    torch.cuda.empty_cache()
    gc.collect()

def scale_tv(tv, alpha):
    """Scala un TaskVector moltiplicando ogni parametro per alpha.
    Supporta alpha negativi per la task negation.
    """
    scaled = copy.deepcopy(tv)
    scaled.vector = {k: v * alpha for k, v in tv.vector.items()}
    return scaled


# ========== CARICAMENTO TASK VECTORS ==========
print("\n🔄 Caricamento Task Vectors (CPU)...")
tv_gtsrb = TaskVector(
    pretrained_checkpoint=pretrained,
    finetuned_checkpoint=f'checkpoints/{model}/GTSRB/finetuned.pt'
)
tv_svhn = TaskVector(
    pretrained_checkpoint=pretrained,
    finetuned_checkpoint=f'checkpoints/{model}/SVHN/finetuned.pt'
)
print("✅ Task Vectors caricati.\n")

# ============================================================
# FASE 0 — BASELINE ZERO-SHOT
# ============================================================
if RUN_BASELINE:
    print_header("FASE 0 — BASELINE ZERO-SHOT",
                 "Nessun task vector (scaling_coef=0) — riferimento assoluto")
    m = tv_gtsrb.apply_to(pretrained, scaling_coef=0.0)
    acc_g, acc_s, mem = eval_model(m)
    print_row("α=0.0", acc_g, acc_s, mem)
    cleanup(m)

# ============================================================
# FASE 1 — SOLO τ_GTSRB × α
# ============================================================
if RUN_ONLY_GTSRB:
    print_header("FASE 1 — SOLO τ_GTSRB × α",
                 "Upper bound GTSRB | SVHN invariato (nessun contributo τ_SVHN)")
    for alpha in alphas:
        m = tv_gtsrb.apply_to(pretrained, scaling_coef=abs(alpha)) \
            if alpha >= 0 else \
            (-tv_gtsrb).apply_to(pretrained, scaling_coef=abs(alpha))
        acc_g, acc_s, mem = eval_model(m)
        print_row(f"α={alpha:+.1f}", acc_g, acc_s, mem)
        cleanup(m)

# ============================================================
# FASE 2 — SOLO τ_SVHN × α
# ============================================================
if RUN_ONLY_SVHN:
    print_header("FASE 2 — SOLO τ_SVHN × α",
                 "Upper bound SVHN | GTSRB invariato (nessun contributo τ_GTSRB)")
    for alpha in alphas:
        m = tv_svhn.apply_to(pretrained, scaling_coef=abs(alpha)) \
            if alpha >= 0 else \
            (-tv_svhn).apply_to(pretrained, scaling_coef=abs(alpha))
        acc_g, acc_s, mem = eval_model(m)
        print_row(f"α={alpha:+.1f}", acc_g, acc_s, mem)
        cleanup(m)

# ============================================================
# FASE 3 — MULTI-TASK (τ_GTSRB + τ_SVHN) × α
# ============================================================
if RUN_MULTI:
    tv_multi = tv_gtsrb + tv_svhn
    print_header("FASE 3 — MULTI-TASK (τ_GTSRB + τ_SVHN) × α",
                 "α unico condiviso | interferenza = Δ rispetto a Fasi 1/2")
    for alpha in alphas:
        m = tv_multi.apply_to(pretrained, scaling_coef=abs(alpha)) \
            if alpha >= 0 else \
            (-tv_multi).apply_to(pretrained, scaling_coef=abs(alpha))
        acc_g, acc_s, mem = eval_model(m)
        print_row(f"α={alpha:+.1f}", acc_g, acc_s, mem)
        cleanup(m)
    del tv_multi
    gc.collect()

# ============================================================
# FASE 4 — PESATO: τ_GTSRB×α1 + τ_SVHN×α2
# Alpha indipendenti per minimizzare l'interferenza in modo asimmetrico.
# Obiettivo: superare simultaneamente GTSRB 96.9% e SVHN 96.4% (Fase 3).
# ============================================================
if RUN_WEIGHTED:
    print_header("FASE 4 — PESATO τ_GTSRB×α1 + τ_SVHN×α2",
                 "α indipendenti | obiettivo: ridurre interferenza vs Fase 3")
    for a1, a2 in WEIGHTED_ALPHAS:
        torch.cuda.empty_cache()
        gc.collect()
        tv_weighted = scale_tv(tv_gtsrb, a1) + scale_tv(tv_svhn, a2)
        m = tv_weighted.apply_to(pretrained, scaling_coef=1.0)
        acc_g, acc_s, mem = eval_model(m)
        print_row(f"({a1:.1f}, {a2:.1f})", acc_g, acc_s, mem)
        cleanup(m)
        del tv_weighted
        gc.collect()

# ============================================================
# FASE 5 — COMBINAZIONE ASIMMETRICA: τ_SVHN + (−τ_GTSRB×α)
# ============================================================
if RUN_NEGATION_GTSRB:
    print_header("FASE 5 — τ_SVHN + (−τ_GTSRB×α)",
                 "Misura l'interferenza di τ_GTSRB su SVHN nel combinato")
    for alpha in NEGATION_ALPHAS:
        torch.cuda.empty_cache()
        gc.collect()
        tv_neg_g = scale_tv(tv_svhn, 1.0) + scale_tv(tv_gtsrb, -alpha)
        m = tv_neg_g.apply_to(pretrained, scaling_coef=1.0)
        acc_g, acc_s, mem = eval_model(m)
        print_row(f"α_neg={alpha:.1f}", acc_g, acc_s, mem)
        cleanup(m)
        del tv_neg_g
        gc.collect()

# ============================================================
# FASE 6 — COMBINAZIONE ASIMMETRICA: τ_GTSRB + (−τ_SVHN×α)
# ============================================================
if RUN_NEGATION_SVHN:
    print_header("FASE 6 — τ_GTSRB + (−τ_SVHN×α)",
                 "Misura l'interferenza di τ_SVHN su GTSRB nel combinato")
    for alpha in NEGATION_ALPHAS:
        torch.cuda.empty_cache()
        gc.collect()
        tv_neg_s = scale_tv(tv_gtsrb, 1.0) + scale_tv(tv_svhn, -alpha)
        m = tv_neg_s.apply_to(pretrained, scaling_coef=1.0)
        acc_g, acc_s, mem = eval_model(m)
        print_row(f"α_neg={alpha:.1f}", acc_g, acc_s, mem)
        cleanup(m)
        del tv_neg_s
        gc.collect()

# ============================================================
# FASE 7 — TASK NEGATION SUL BASE MODEL
# ============================================================
if RUN_NEGATION_BASE:

    # 7a — neghiamo τ_GTSRB sul base model
    print_header("FASE 7a — NEGAZIONE τ_GTSRB sul BASE model",
                 "pretrained − α·τ_GTSRB | atteso: GTSRB << 43.3%")
    for alpha in NEGATION_ALPHAS:
        torch.cuda.empty_cache()
        gc.collect()
        tv_neg = scale_tv(tv_gtsrb, -alpha)
        m = tv_neg.apply_to(pretrained, scaling_coef=1.0)
        acc_g, acc_s, mem = eval_model(m)
        print_row(f"α_neg={alpha:.1f}", acc_g, acc_s, mem)
        cleanup(m)
        del tv_neg
        gc.collect()

    # 7b — neghiamo τ_SVHN sul base model
    print_header("FASE 7b — NEGAZIONE τ_SVHN sul BASE model",
                 "pretrained − α·τ_SVHN | atteso: SVHN << 52.0%")
    for alpha in NEGATION_ALPHAS:
        torch.cuda.empty_cache()
        gc.collect()
        tv_neg = scale_tv(tv_svhn, -alpha)
        m = tv_neg.apply_to(pretrained, scaling_coef=1.0)
        acc_g, acc_s, mem = eval_model(m)
        print_row(f"α_neg={alpha:.1f}", acc_g, acc_s, mem)
        cleanup(m)
        del tv_neg
        gc.collect()

# ============================================================
# FASE 8 — TASK NEGATION SUL MULTI-TASK MODEL  
# ============================================================
if RUN_NEGATION_MULTITASK:

    # 8a — neghiamo τ_GTSRB nel multi-task model (τ_SVHN rimane)
    print_header("FASE 8a — NEGAZIONE τ_GTSRB nel MULTI-TASK model",
                 "pretrained + τ_SVHN + (1−α)·τ_GTSRB | SVHN deve restare ~97%")
    for alpha in NEGATION_ALPHAS:
        torch.cuda.empty_cache()
        gc.collect()
        # α=1.0 → solo τ_SVHN → sanity check con Fase 2
        # α>1.0 → negazione attiva di τ_GTSRB
        tv_neg = scale_tv(tv_svhn, 1.0) + scale_tv(tv_gtsrb, 1.0 - alpha)
        m = tv_neg.apply_to(pretrained, scaling_coef=1.0)
        acc_g, acc_s, mem = eval_model(m)
        print_row(f"α_neg={alpha:.1f}", acc_g, acc_s, mem)
        cleanup(m)
        del tv_neg
        gc.collect()

    # 8b — neghiamo τ_SVHN nel multi-task model (τ_GTSRB rimane)
    print_header("FASE 8b — NEGAZIONE τ_SVHN nel MULTI-TASK model",
                 "pretrained + τ_GTSRB + (1−α)·τ_SVHN | GTSRB deve restare ~99%")
    for alpha in NEGATION_ALPHAS:
        torch.cuda.empty_cache()
        gc.collect()
        # α=1.0 → solo τ_GTSRB → sanity check con Fase 1
        # α>1.0 → negazione attiva di τ_SVHN
        tv_neg = scale_tv(tv_gtsrb, 1.0) + scale_tv(tv_svhn, 1.0 - alpha)
        m = tv_neg.apply_to(pretrained, scaling_coef=1.0)
        acc_g, acc_s, mem = eval_model(m)
        print_row(f"α_neg={alpha:.1f}", acc_g, acc_s, mem)
        cleanup(m)
        del tv_neg
        gc.collect()

# ============================================================
print("\n" + "="*60)
print("  ✅ ANALISI COMPLETA — 8 FASI")
print()
print("  Fase 0      → baseline zero-shot (riferimento assoluto)")
print("  Fasi 1/2    → upper bound task singolo (no interferenza)")
print("  Fase 3      → multi-task α unico → misuri l'interferenza")
print("  Fase 4      → α indipendenti → tenti di ridurla")
print("  Fasi 5/6    → combinazioni asimmetriche (analisi interferenza)")
print("  Fase 7      → negazione sul BASE model → valida il task vector")
print("  Fase 8      → negazione sul MULTI-TASK → selettività")
print()
print("="*60)