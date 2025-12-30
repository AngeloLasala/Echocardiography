"""
Compute model complexity of regression model - Parameters, GFLOPS, and inference time
"""
import torch
import json
from pathlib import Path
import time
import numpy as np
from echocardiography.regression.cfg import train_config
from echocardiography.regression.utils import get_corrdinate_from_heatmap, get_corrdinate_from_heatmap_ellipses, echocardiografic_parameters
from PIL import Image
import copy

# Load model
model_name = 'unet_up'
method = 'max_value'  # 'max_value' o 'ellipses'
cfg = train_config('heatmaps', 
                    threshold_wloss=0.5, 
                    model=model_name,
                    input_channels=1,                       
                    device='cuda')

# Estrai il modello dalla configurazione
model = cfg['model']

# Parametri per il test
input_channels = cfg.get('input_channels', 1)
input_height = 256  # Modifica in base alle dimensioni delle tue immagini
input_width = 256   # Modifica in base alle dimensioni delle tue immagini
num_iterations = 100
warmup_iterations = 10

# ============================================================================
# Funzioni per calcolo FLOPs
# ============================================================================

def count_conv2d_flops(module, input, output):
    """Calcola FLOPs per layer Conv2d"""
    input = input[0]
    batch_size = input.shape[0]
    output_height, output_width = output.shape[2:]
    
    kernel_height, kernel_width = module.kernel_size
    in_channels = module.in_channels
    out_channels = module.out_channels
    groups = module.groups
    
    flops = batch_size * output_height * output_width * \
            (in_channels / groups) * out_channels * \
            kernel_height * kernel_width * 2
    
    module.__flops__ += int(flops)

def count_linear_flops(module, input, output):
    """Calcola FLOPs per layer Linear"""
    input = input[0]
    batch_size = input.shape[0]
    in_features = module.in_features
    out_features = module.out_features
    
    # FLOPs = 2 * batch_size * in_features * out_features
    flops = 2 * batch_size * in_features * out_features
    module.__flops__ += int(flops)

def count_normalization_flops(module, input, output):
    """Calcola FLOPs per normalization layers"""
    input = input[0]
    # Normalization: mean, variance, normalization (3 ops per element)
    flops = input.numel() * 3
    module.__flops__ += int(flops)

def count_activation_flops(module, input, output):
    """Calcola FLOPs per activation functions"""
    # 1 operazione per elemento
    flops = output.numel()
    module.__flops__ += int(flops)

def calculate_flops(model, input_shape, device):
    """Calcola i GFLOPs totali del modello"""
    # Usa il modello direttamente invece di creare una copia
    model = model.to(device)
    model.eval()
    
    # Rimuovi eventuali hooks esistenti
    hooks = []
    
    # Inizializza contatori
    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        
        m.__flops__ = 0
        
        handle = None
        if isinstance(m, torch.nn.Conv2d):
            handle = m.register_forward_hook(count_conv2d_flops)
        elif isinstance(m, torch.nn.Linear):
            handle = m.register_forward_hook(count_linear_flops)
        elif isinstance(m, (torch.nn.InstanceNorm2d, torch.nn.BatchNorm2d, 
                           torch.nn.LayerNorm, torch.nn.GroupNorm)):
            handle = m.register_forward_hook(count_normalization_flops)
        elif isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.PReLU,
                           torch.nn.ReLU6, torch.nn.ELU, torch.nn.GELU,
                           torch.nn.SiLU, torch.nn.Sigmoid, torch.nn.Tanh)):
            handle = m.register_forward_hook(count_activation_flops)
        
        if handle is not None:
            hooks.append(handle)
    
    model.apply(add_hooks)
    
    # Forward pass
    input_tensor = torch.randn(input_shape).to(device)
    with torch.no_grad():
        try:
            model(input_tensor)
        except Exception as e:
            print(f"Errore durante il calcolo dei FLOPs: {e}")
            # Rimuovi gli hooks anche in caso di errore
            for hook in hooks:
                hook.remove()
            return 0
    
    # Somma tutti i FLOPs
    total_flops = 0
    for m in model.modules():
        if hasattr(m, '__flops__'):
            total_flops += m.__flops__
            delattr(m, '__flops__')  # Pulisci l'attributo
    
    # Rimuovi gli hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops

def benchmark_inference(model, input_shape, device, num_iterations=100, warmup=10):
    """Esegue benchmark del tempo di inferenza"""
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warm-up
    print(f"  Warming up on {device}...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Sincronizza
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Misura tempo di inferenza
    print(f"  Measuring inference time on {device}...")
    inference_times = []
    
    for _ in range(num_iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model(dummy_input).cpu().numpy()[0]
            output_res = [Image.fromarray(output[ch,:,:] * 255) for ch in range(output.shape[0])]
            output = np.array(output_res)


            if method == 'max_value': output = get_corrdinate_from_heatmap(output)
            if method == 'ellipses': output = get_corrdinate_from_heatmap_ellipses(output)
            
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000)  # ms
    
    # Calcola statistiche
    results = {
        'mean': np.mean(inference_times),
        'std': np.std(inference_times),
        'min': np.min(inference_times),
        'max': np.max(inference_times),
        'median': np.median(inference_times),
        'fps': 1000 / np.mean(inference_times),
        # 'output_shape': list(output.shape) if isinstance(output, torch.Tensor) else [list(o.shape) for o in output]
    }
    
    return results

# ============================================================================
# Calcolo delle metriche
# ============================================================================

# Calcola parametri
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Input shape
input_shape = (1, input_channels, input_height, input_width)

# Calcola GFLOPs
print("\nCalcolando GFLOPs...")
try:
    total_flops = calculate_flops(model, input_shape, torch.device('cpu'))
    gflops = total_flops / 1e9
    print(f"Calcolo completato: {gflops:.2f} GFLOPs")
except Exception as e:
    print(f"Errore nel calcolo dei FLOPs: {e}")
    gflops = None

# ============================================================================
# Stampa risultati generali
# ============================================================================

print(f"\n{'='*70}")
print(f"Statistiche del modello - {model_name}")
print(f"{'='*70}")
print(f"Model:               {model_name}")
print(f"Input shape:         {list(input_shape)}")
print(f"-" * 70)
print(f"Input channels:      {input_channels}")
print(f"Input size:          {input_height}x{input_width}")
print(f"-" * 70)
print(f"Parametri totali:    {total_params / 1e6:.2f} M ({total_params:,})")
print(f"Parametri trainable: {trainable_params / 1e6:.2f} M ({trainable_params:,})")
if gflops is not None:
    print(f"GFLOPs:              {gflops:.2f}")
else:
    print(f"GFLOPs:              N/A (errore nel calcolo)")
print(f"{'='*70}\n")

# ============================================================================
# Benchmark su CPU
# ============================================================================

print("=" * 70)
print("BENCHMARK CPU")
print("=" * 70)
cpu_results = benchmark_inference(model, input_shape, torch.device('cpu'), 
                                  num_iterations=num_iterations, 
                                  warmup=warmup_iterations)
# print(f"Output shape:        {cpu_results['output_shape']}")
print(f"-" * 70)
print(f"Tempo di inferenza ({num_iterations} iterations):")
print(f"  Mean:              {cpu_results['mean']:.2f} ms")
print(f"  Median:            {cpu_results['median']:.2f} ms")
print(f"  Std:               {cpu_results['std']:.2f} ms")
print(f"  Min:               {cpu_results['min']:.2f} ms")
print(f"  Max:               {cpu_results['max']:.2f} ms")
print(f"  FPS:               {cpu_results['fps']:.2f}")
print(f"{'='*70}\n")

# ============================================================================
# Benchmark su GPU (se disponibile)
# ============================================================================

if torch.cuda.is_available():
    print("=" * 70)
    print(f"BENCHMARK GPU ({torch.cuda.get_device_name(0)})")
    print("=" * 70)
    gpu_results = benchmark_inference(model, input_shape, torch.device('cuda'), 
                                      num_iterations=num_iterations, 
                                      warmup=warmup_iterations)
    # print(f"Output shape:        {gpu_results['output_shape']}")
    print(f"-" * 70)
    print(f"Tempo di inferenza ({num_iterations} iterations):")
    print(f"  Mean:              {gpu_results['mean']:.2f} ms")
    print(f"  Median:            {gpu_results['median']:.2f} ms")
    print(f"  Std:               {gpu_results['std']:.2f} ms")
    print(f"  Min:               {gpu_results['min']:.2f} ms")
    print(f"  Max:               {gpu_results['max']:.2f} ms")
    print(f"  FPS:               {gpu_results['fps']:.2f}")
    print(f"-" * 70)
    print(f"Speedup GPU vs CPU: {cpu_results['mean'] / gpu_results['mean']:.2f}x")
    print(f"{'='*70}\n")
else:
    print("GPU non disponibile. Benchmark solo su CPU.\n")
    gpu_results = None

# ============================================================================
# Salva risultati su file
# ============================================================================

results_summary = {
    'model_name': model_name,
    'input_shape': list(input_shape),
    'total_params': int(total_params),
    'trainable_params': int(trainable_params),
    'params_M': round(total_params / 1e6, 2),
    'gflops': float(gflops) if gflops is not None else None,
    'cpu': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
            for k, v in cpu_results.items()}
}

if gpu_results is not None:
    results_summary['gpu'] = {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in gpu_results.items()}
    results_summary['speedup'] = float(cpu_results['mean'] / gpu_results['mean'])
    results_summary['gpu_name'] = torch.cuda.get_device_name(0)

output_file = f'model_complexity_{model_name}.json'
with open(output_file, 'w') as f:
    json.dump(results_summary, f, indent=4)

print(f"Risultati salvati in: {output_file}")

# ============================================================================
# Stampa riepilogo finale
# ============================================================================

print(f"\n{'='*70}")
print(f"RIEPILOGO FINALE")
print(f"{'='*70}")
print(f"Modello:             {model_name}")
print(f"Parametri:           {total_params / 1e6:.2f} M")
if gflops is not None:
    print(f"GFLOPs:              {gflops:.2f}")
print(f"CPU FPS:             {cpu_results['fps']:.2f}")
if gpu_results is not None:
    print(f"GPU FPS:             {gpu_results['fps']:.2f}")
    print(f"Speedup:             {cpu_results['mean'] / gpu_results['mean']:.2f}x")
print(f"{'='*70}")