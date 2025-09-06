# Cherry-picked some good parts from ComfyUI with some bad parts fixed

import sys
import time
import psutil
import torch
import platform
import os

from enum import Enum
from . import stream, utils    # este archivo vive en el paquete backend
from backend.args import args   # args globales de Forge

# --- Forge compatibility stubs: bitsandbytes (BNB) en DirectML/AMD ---
# En Windows + DirectML no usamos bitsandbytes; exponemos símbolos esperados.
BNB_AVAILABLE = False

def can_install_bnb() -> bool:
    """
    Forge llama a esto para decidir si intenta instalar bitsandbytes.
    En DirectML/AMD debe devolver False para que NO intente instalarlo.
    """
    return False

def is_bitsandbytes_available() -> bool:
    """Compat: algunos módulos chequean disponibilidad explícita."""
    return BNB_AVAILABLE

def ensure_bitsandbytes() -> bool:
    """
    Compat: si algún módulo intenta 'asegurar' BNB, devolvemos False
    para indicar que no hay nada que hacer/instalar.
    """
    return False
# ---------------------------------------------------------------------

cpu = torch.device('cpu')


class VRAMState(Enum):
    DISABLED = 0  # No vram present: no need to move models to vram
    NO_VRAM = 1  # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5  # No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

lowvram_available = True
xpu_available = False

if getattr(args, "pytorch_deterministic", False):
    print("Using deterministic algorithms for pytorch")
    torch.use_deterministic_algorithms(True, warn_only=True)

# ------- DirectML detection (supports --use-directml o --directml N) -------
directml_enabled = False
directml_device = None
_directml_idx = None
if getattr(args, "use_directml", False) or getattr(args, "directml", None) is not None:
    try:
        import torch_directml  # Optional dependency: pip install torch-directml
        directml_enabled = True
        if getattr(args, "directml", None) is not None:
            _directml_idx = args.directml
        else:
            _directml_idx = -1
        if _directml_idx is not None and _directml_idx >= 0:
            directml_device = torch_directml.device(_directml_idx)
            print(f"Using DirectML with device index: {_directml_idx} ({torch_directml.device_name(_directml_idx)})")
        else:
            directml_device = torch_directml.device()
            print("Using DirectML with default device")
    except Exception as e:
        print(f"WARNING: DirectML requested but torch_directml not available: {e}")
        directml_enabled = False
# ---------------------------------------------------------------------------

try:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            import intel_extension_for_pytorch  # Only import if available
            xpu_available = True
        except ImportError:
            xpu_available = False
except Exception:
    pass

try:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except Exception:
    pass

if getattr(args, "always_cpu", False):
    cpu_state = CPUState.CPU


def is_intel_xpu():
    global cpu_state
    global xpu_available
    return cpu_state == CPUState.GPU and xpu_available


def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled and directml_device is not None:
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    # GPU sin DirectML: sólo si CUDA está disponible
    if torch.cuda.is_available():
        try:
            return torch.device(torch.cuda.current_device())
        except Exception:
            return torch.device("cpu")
    # Intel XPU
    if is_intel_xpu():
        try:
            return torch.device("xpu", torch.xpu.current_device())
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def get_total_memory(dev=None, torch_total_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    # CPU/MPS
    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        # DirectML: sin API directa -> devolver valor seguro (ajustable)
        if directml_enabled:
            # Si sabés la VRAM exacta de tu GPU, podés ajustarla aquí:
            mem_total = 8 * 1024 * 1024 * 1024  # 8 GB en bytes
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats.get('reserved_bytes.all.current', 0)
            mem_total_torch = mem_reserved
            mem_total = torch.xpu.get_device_properties(dev).total_memory
        elif torch.cuda.is_available():
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats.get('reserved_bytes.all.current', 0)
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda
        else:
            # Sin backend conocido
            mem_total = psutil.virtual_memory().total
            mem_total_torch = mem_total

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total


total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
print("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

try:
    print("pytorch version: {}".format(torch.version.__version__))
except Exception:
    pass

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except Exception:
    OOM_EXCEPTION = Exception

if directml_enabled:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if getattr(args, "disable_xformers", False):
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops
        XFORMERS_IS_AVAILABLE = True
        try:
            XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
        except Exception:
            pass
        try:
            XFORMERS_VERSION = xformers.version.__version__
            print("xformers version: {}".format(XFORMERS_VERSION))
            if XFORMERS_VERSION.startswith("0.0.18"):
                print("\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
                print("Please downgrade or upgrade xformers to a different version.\n")
                XFORMERS_ENABLED_VAE = False
        except Exception:
            pass
    except Exception:
        # En DirectML o sin CUDA, xformers no aplica
        XFORMERS_IS_AVAILABLE = False


def is_nvidia():
    return bool(getattr(torch, "version", None) and getattr(torch.version, "cuda", None))


ENABLE_PYTORCH_ATTENTION = False
if getattr(args, "attention_pytorch", False):
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False

VAE_DTYPES = [torch.float32]

try:
    if is_nvidia():
        torch_version = torch.version.__version__
        if int(torch_version.split('.')[0]) >= 2:
            if not ENABLE_PYTORCH_ATTENTION and not getattr(args, "attention_split", False) and not getattr(args, "attention_quad", False):
                ENABLE_PYTORCH_ATTENTION = True
            if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8:
                    VAE_DTYPES = [torch.bfloat16] + VAE_DTYPES
    if is_intel_xpu():
        if not getattr(args, "attention_split", False) and not getattr(args, "attention_quad", False):
            ENABLE_PYTORCH_ATTENTION = True
except Exception:
    pass

if is_intel_xpu():
    VAE_DTYPES = [torch.bfloat16] + VAE_DTYPES

if getattr(args, "vae_in_cpu", False):
    VAE_DTYPES = [torch.float32]

VAE_ALWAYS_TILED = False

# Activa PyTorch SDP sólo si hay backend CUDA real
if ENABLE_PYTORCH_ATTENTION and hasattr(torch, "backends") and hasattr(torch.backends, "cuda") and torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

if getattr(args, "always_low_vram", False):
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif getattr(args, "always_no_vram", False):
    set_vram_to = VRAMState.NO_VRAM
elif getattr(args, "always_high_vram", False) or getattr(args, "always_gpu", False):
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = bool(getattr(args, "all_in_fp32", False))
FORCE_FP16 = bool(getattr(args, "all_in_fp16", False))
if FORCE_FP32:
    print("Forcing FP32, if this improves things please report it.")
if FORCE_FP16:
    print("Forcing FP16.")

if lowvram_available and set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
    vram_state = set_vram_to

if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

print(f"Set vram state to: {vram_state.name}")

ALWAYS_VRAM_OFFLOAD = bool(getattr(args, "always_offload_from_vram", False))
if ALWAYS_VRAM_OFFLOAD:
    print("Always offload VRAM")

PIN_SHARED_MEMORY = bool(getattr(args, "pin_shared_memory", False))
if PIN_SHARED_MEMORY:
    print("Always pin shared GPU memory")


def get_torch_device_name(device):
    try:
        if hasattr(device, 'type'):
            if device.type == "cuda":
                try:
                    allocator_backend = torch.cuda.get_allocator_backend()
                except Exception:
                    allocator_backend = ""
                return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
            elif device.type == "privateuseone":
                return "DirectML (privateuseone)"
            else:
                return "{}".format(device.type)
        elif is_intel_xpu():
            return "{} {}".format(device, torch.xpu.get_device_name(device))
        else:
            return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))
    except Exception:
        return str(device)


try:
    torch_device_name = get_torch_device_name(get_torch_device())
    print("Device: {}".format(torch_device_name))
except Exception:
    torch_device_name = ''
    print("Could not pick default device.")

if 'rtx' in torch_device_name.lower():
    if not getattr(args, "cuda_malloc", False):
        print('Hint: your device supports --cuda-malloc for potential speed improvements.')


current_loaded_models = []


def state_dict_size(sd, exclude_device=None):
    module_mem = 0
    for k in sd:
        t = sd[k]
        if exclude_device is not None and t.device == exclude_device:
            continue
        module_mem += t.nelement() * t.element_size()
    return module_mem


def state_dict_parameters(sd):
    module_mem = 0
    for _, v in sd.items():
        module_mem += v.nelement()
    return module_mem


def state_dict_dtype(state_dict):
    for k, v in state_dict.items():
        if hasattr(v, 'gguf_cls'):
            return 'gguf'
        if 'bitsandbytes__nf4' in k:
            return 'nf4'
        if 'bitsandbytes__fp4' in k:
            return 'fp4'

    dtype_counts = {}
    for tensor in state_dict.values():
        dtype = tensor.dtype
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

    major_dtype = None
    max_count = 0
    for dtype, count in dtype_counts.items():
        if count > max_count:
            max_count = count
            major_dtype = dtype
    return major_dtype


def bake_gguf_model(model):
    if getattr(model, 'gguf_baked', False):
        return
    for p in model.parameters():
        gguf_cls = getattr(p, 'gguf_cls', None)
        if gguf_cls is not None:
            gguf_cls.bake(p)
    global signal_empty_cache
    signal_empty_cache = True
    model.gguf_baked = True
    return model


def module_size(module, exclude_device=None, include_device=None, return_split=False):
    module_mem = 0
    weight_mem = 0
    weight_patterns = ['weight']

    for k, p in module.named_parameters():
        t = p.data
        if exclude_device is not None and t.device == exclude_device:
            continue
        if include_device is not None and t.device != include_device:
            continue

        element_size = t.element_size()
        if getattr(p, 'quant_type', None) in ['fp4', 'nf4']:
            if element_size > 1:
                element_size = 0.55
            else:
                element_size = 1.1

        module_mem += t.nelement() * element_size
        if k in weight_patterns:
            weight_mem += t.nelement() * element_size

    if return_split:
        return module_mem, weight_mem, module_mem - weight_mem
    return module_mem


def module_move(module, device, recursive=True, excluded_pattens=[]):
    if recursive:
        return module.to(device=device)
    for k, p in module.named_parameters(recurse=False, remove_duplicate=True):
        if k in excluded_pattens:
            continue
        setattr(module, k, utils.tensor2parameter(p.to(device=device)))
    return module


def build_module_profile(model, model_gpu_memory_when_using_cpu_swap):
    all_modules = []
    legacy_modules = []

    for m in model.modules():
        if hasattr(m, "parameters_manual_cast"):
            m.total_mem, m.weight_mem, m.extra_mem = module_size(m, return_split=True)
            all_modules.append(m)
        elif hasattr(m, "weight"):
            m.total_mem, m.weight_mem, m.extra_mem = module_size(m, return_split=True)
            legacy_modules.append(m)

    gpu_modules = []
    gpu_modules_only_extras = []
    mem_counter = 0

    for m in legacy_modules.copy():
        gpu_modules.append(m)
        legacy_modules.remove(m)
        mem_counter += m.total_mem

    for m in sorted(all_modules, key=lambda x: x.extra_mem).copy():
        if mem_counter + m.extra_mem < model_gpu_memory_when_using_cpu_swap:
            gpu_modules_only_extras.append(m)
            all_modules.remove(m)
            mem_counter += m.extra_mem

    cpu_modules = all_modules

    for m in sorted(gpu_modules_only_extras, key=lambda x: x.weight_mem).copy():
        if mem_counter + m.weight_mem < model_gpu_memory_when_using_cpu_swap:
            gpu_modules.append(m)
            gpu_modules_only_extras.remove(m)
            mem_counter += m.weight_mem

    return gpu_modules, gpu_modules_only_extras, cpu_modules


def _should_use_stream():
    try:
        return bool(stream.should_use_stream())
    except Exception:
        return False


class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.model_accelerated = False
        self.device = model.load_device
        self.inclusive_memory = 0
        self.exclusive_memory = 0

    def compute_inclusive_exclusive_memory(self):
        self.inclusive_memory = module_size(self.model.model, include_device=self.device)
        self.exclusive_memory = module_size(self.model.model, exclude_device=self.device)
        return

    def model_load(self, model_gpu_memory_when_using_cpu_swap=-1):
        patch_model_to = None
        do_not_need_cpu_swap = model_gpu_memory_when_using_cpu_swap < 0

        if do_not_need_cpu_swap:
            patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        try:
            self.real_model = self.model.forge_patch_model(patch_model_to)
            self.model.current_device = self.model.load_device
        except Exception as e:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e

        if do_not_need_cpu_swap:
            print('All loaded to GPU.')
        else:
            gpu_modules, gpu_modules_only_extras, cpu_modules = build_module_profile(self.real_model, model_gpu_memory_when_using_cpu_swap)
            pin_memory = PIN_SHARED_MEMORY and is_device_cpu(self.model.offload_device)

            mem_counter = 0
            swap_counter = 0

            for m in gpu_modules:
                m.to(self.device)
                mem_counter += m.total_mem

            for m in cpu_modules:
                m.prev_parameters_manual_cast = m.parameters_manual_cast
                m.parameters_manual_cast = True
                m.to(self.model.offload_device)
                if pin_memory:
                    m._apply(lambda x: x.pin_memory())
                swap_counter += m.total_mem

            for m in gpu_modules_only_extras:
                m.prev_parameters_manual_cast = m.parameters_manual_cast
                m.parameters_manual_cast = True
                module_move(m, device=self.device, recursive=False, excluded_pattens=['weight'])
                if hasattr(m, 'weight') and m.weight is not None:
                    if pin_memory:
                        m.weight = utils.tensor2parameter(m.weight.to(self.model.offload_device).pin_memory())
                    else:
                        m.weight = utils.tensor2parameter(m.weight.to(self.model.offload_device))
                mem_counter += m.extra_mem
                swap_counter += m.weight_mem

            swap_flag = 'Shared' if PIN_SHARED_MEMORY else 'CPU'
            method_flag = 'asynchronous' if _should_use_stream() else 'blocked'
            print(f"{swap_flag} Swap Loaded ({method_flag} method): {swap_counter / (1024 * 1024):.2f} MB, GPU Loaded: {mem_counter / (1024 * 1024):.2f} MB")

            self.model_accelerated = True

            global signal_empty_cache
            signal_empty_cache = True

        bake_gguf_model(self.real_model)

        self.model.refresh_loras()

        if is_intel_xpu() and not getattr(args, "disable_ipex_hijack", False):
            self.real_model = torch.xpu.optimize(self.real_model.eval(), inplace=True, auto_kernel_selection=True, graph_mode=True)

        return self.real_model

    def model_unload(self, avoid_model_moving=False):
        if self.model_accelerated:
            for m in self.real_model.modules():
                if hasattr(m, "prev_parameters_manual_cast"):
                    m.parameters_manual_cast = m.prev_parameters_manual_cast
                    del m.prev_parameters_manual_cast
            self.model_accelerated = False

        if avoid_model_moving:
            self.model.forge_unpatch_model()
        else:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model.model_patches_to(self.model.offload_device)

    def __eq__(self, other):
        return self.model is other.model


# --------- OVERRIDE: inference_memory desde env (MB) ----------
try:
    _im_mb = float(os.environ.get("inference_memory", "1024"))
except Exception:
    _im_mb = 1024.0
current_inference_memory = int(_im_mb * 1024 * 1024)
print(f"[GPU Setting Override] inference_memory = {_im_mb:.0f} MB")
# --------------------------------------------------------------


def minimum_inference_memory():
    global current_inference_memory
    return current_inference_memory


def unload_model_clones(model):
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload
    for i in to_unload:
        current_loaded_models.pop(i).model_unload(avoid_model_moving=True)


def free_memory(memory_required, device, keep_loaded=[], free_all=False):
    # Unload abandonados
    for i in range(len(current_loaded_models) - 1, -1, -1):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            current_loaded_models.pop(i).model_unload(avoid_model_moving=True)

    if free_all:
        memory_required = 1e30
        print(f"[Unload] Trying to free all memory for {device} with {len(keep_loaded)} models keep loaded ... ", end="")
    else:
        print(f"[Unload] Trying to free {memory_required / (1024 * 1024):.2f} MB for {device} with {len(keep_loaded)} models keep loaded ... ", end="")

    offload_everything = ALWAYS_VRAM_OFFLOAD or vram_state == VRAMState.NO_VRAM
    unloaded_model = False
    for i in range(len(current_loaded_models) - 1, -1, -1):
        if not offload_everything:
            free_mem = get_free_memory(device)
            print(f"Current free memory is {free_mem / (1024 * 1024):.2f} MB ... ", end="")
            if free_mem > memory_required:
                break
        shift_model = current_loaded_models[i]
        if shift_model.device == device and shift_model not in keep_loaded:
            m = current_loaded_models.pop(i)
            print(f"Unload model {m.model.model.__class__.__name__} ", end="")
            m.model_unload()
            del m
            unloaded_model = True

    if unloaded_model:
        soft_empty_cache()
    else:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()

    print('Done.')
    return


def compute_model_gpu_memory_when_using_cpu_swap(current_free_mem, inference_memory):
    maximum_memory_available = current_free_mem - inference_memory
    suggestion = max(
        maximum_memory_available / 1.3,
        maximum_memory_available - 1024 * 1024 * 1024 * 1.25
    )
    return int(max(0, suggestion))


def load_models_gpu(models, memory_required=0, hard_memory_preservation=0):
    global vram_state

    execution_start_time = time.perf_counter()
    memory_to_free = max(minimum_inference_memory(), memory_required) + hard_memory_preservation
    memory_for_inference = minimum_inference_memory() + hard_memory_preservation

    models_to_load = []
    models_already_loaded = []
    for x in models:
        loaded_model = LoadedModel(x)
        if loaded_model in current_loaded_models:
            index = current_loaded_models.index(loaded_model)
            current_loaded_models.insert(0, current_loaded_models.pop(index))
            models_already_loaded.append(loaded_model)
        else:
            models_to_load.append(loaded_model)

    if len(models_to_load) == 0:
        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_memory(memory_to_free, d, models_already_loaded)
        moving_time = time.perf_counter() - execution_start_time
        if moving_time > 0.1:
            print(f'Memory cleanup has taken {moving_time:.2f} seconds')
        return

    for loaded_model in models_to_load:
        unload_model_clones(loaded_model.model)

    total_memory_required = {}
    for loaded_model in models_to_load:
        loaded_model.compute_inclusive_exclusive_memory()
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.exclusive_memory + loaded_model.inclusive_memory * 0.25

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.3 + memory_to_free, device, models_already_loaded)

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state

        model_gpu_memory_when_using_cpu_swap = -1

        if lowvram_available and (vram_set_state == VRAMState.LOW_VRAM or vram_set_state == VRAMState.NORMAL_VRAM):
            model_require = loaded_model.exclusive_memory
            previously_loaded = loaded_model.inclusive_memory
            current_free_mem = get_free_memory(torch_dev)
            estimated_remaining_memory = current_free_mem - model_require - memory_for_inference

            print(f"[Memory Management] Target: {loaded_model.model.model.__class__.__name__}, Free GPU: {current_free_mem / (1024 * 1024):.2f} MB, Model Require: {model_require / (1024 * 1024):.2f} MB, Previously Loaded: {previously_loaded / (1024 * 1024):.2f} MB, Inference Require: {memory_for_inference / (1024 * 1024):.2f} MB, Remaining: {estimated_remaining_memory / (1024 * 1024):.2f} MB, ", end="")

            if estimated_remaining_memory < 0:
                vram_set_state = VRAMState.LOW_VRAM
                model_gpu_memory_when_using_cpu_swap = compute_model_gpu_memory_when_using_cpu_swap(current_free_mem, memory_for_inference)
                if previously_loaded > 0:
                    model_gpu_memory_when_using_cpu_swap = previously_loaded

        if vram_set_state == VRAMState.NO_VRAM:
            model_gpu_memory_when_using_cpu_swap = 0

        loaded_model.model_load(model_gpu_memory_when_using_cpu_swap)
        current_loaded_models.insert(0, loaded_model)

    moving_time = time.perf_counter() - execution_start_time
    print(f'Moving model(s) has taken {moving_time:.2f} seconds')
    return


def load_model_gpu(model):
    return load_models_gpu([model])


def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            to_delete = [i] + to_delete
    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x


def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except Exception:
            pass
    return dtype_size


def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")


def unet_inital_load_device(parameters, dtype):
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM:
        return torch_dev
    cpu_dev = torch.device("cpu")
    if ALWAYS_VRAM_OFFLOAD:
        return cpu_dev
    model_size = dtype_size(dtype) * parameters
    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    global directml_enabled

    if device is not None and is_device_cpu(device):
        return False
    if FORCE_FP16:
        return True
    if device is not None and is_device_mps(device):
        return True
    if FORCE_FP32:
        return False
    if directml_enabled:
        return False
    if mps_mode():
        return True
    if cpu_mode():
        return False
    if is_intel_xpu():
        return True
    if getattr(torch, "version", None) and getattr(torch.version, "hip", None):
        return True

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device("cuda"))
        if props.major >= 8:
            return True
        if props.major < 6:
            return False
        nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050", "p40", "p100", "p6", "p4"]
        for x in nvidia_10_series:
            if x in props.name.lower():
                if manual_cast:
                    free_model_memory = (get_free_memory() * 0.9 - minimum_inference_memory())
                    if (not prioritize_performance) or model_params * 4 > free_model_memory:
                        return True
                else:
                    return False
        if props.major < 7:
            return False
        nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"]
        for x in nvidia_16_series:
            if x in props.name:
                return False
        return True

    return False


def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None and is_device_cpu(device):
        return False
    if device is not None and is_device_mps(device):
        return True
    if FORCE_FP32:
        return False
    if directml_enabled:
        return False
    if mps_mode():
        return True
    if cpu_mode():
        return False
    if is_intel_xpu():
        return True

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device("cuda"))
        if props.major >= 8:
            return True
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            if manual_cast:
                free_model_memory = (get_free_memory() * 0.9 - minimum_inference_memory())
                if (not prioritize_performance) or model_params * 4 > free_model_memory:
                    return True
    return False


def get_computation_dtype(inference_device, parameters=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    for candidate in supported_dtypes:
        if candidate == torch.float16 and should_use_fp16(inference_device, model_params=parameters, prioritize_performance=True, manual_cast=False):
            return candidate
        if candidate == torch.bfloat16 and should_use_bf16(inference_device, model_params=parameters, prioritize_performance=True, manual_cast=False):
            return candidate
    return torch.float32


def text_encoder_offload_device():
    if getattr(args, "always_gpu", False):
        return get_torch_device()
    else:
        return torch.device("cpu")


def text_encoder_device():
    if getattr(args, "always_gpu", False):
        return get_torch_device()
    elif vram_state in (VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM):
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def text_encoder_dtype(device=None):
    if getattr(args, "clip_in_fp8_e4m3fn", False):
        return torch.float8_e4m3fn
    elif getattr(args, "clip_in_fp8_e5m2", False):
        return torch.float8_e5m2
    elif getattr(args, "clip_in_fp16", False):
        return torch.float16
    elif getattr(args, "clip_in_fp32", False):
        return torch.float32
    if is_device_cpu(device):
        # En CPU puro conviene FP32 para evitar cast y errores en DirectML
        return torch.float32
    return torch.float16


def intermediate_device():
    if getattr(args, "always_gpu", False):
        return get_torch_device()
    else:
        return torch.device("cpu")


def vae_device():
    if getattr(args, "vae_in_cpu", False):
        return torch.device("cpu")
    return get_torch_device()


def vae_offload_device():
    if getattr(args, "always_gpu", False):
        return get_torch_device()
    else:
        return torch.device("cpu")


def vae_dtype(device=None, allowed_dtypes=[]):
    global VAE_DTYPES
    if getattr(args, "vae_in_fp16", False):
        return torch.float16
    elif getattr(args, "vae_in_bf16", False):
        return torch.bfloat16
    elif getattr(args, "vae_in_fp32", False):
        return torch.float32

    for d in allowed_dtypes:
        if d == torch.float16 and should_use_fp16(device, prioritize_performance=False):
            return d
        if d in VAE_DTYPES:
            return d
    return VAE_DTYPES[0]


print(f"VAE dtype preferences: {VAE_DTYPES} -> {vae_dtype()}")


def get_autocast_device(dev):
    if hasattr(dev, 'type'):
        return dev.type
    return "cuda"


def supports_dtype(device, dtype):  # TODO
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype in (torch.float16, torch.bfloat16):
        return True
    return False


def supports_cast(device, dtype):  # TODO
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if directml_enabled:  # DirectML: evitar cast complejo a tipos "exóticos"
        return False
    if dtype == torch.bfloat16:
        return True
    if is_device_mps(device):
        return False
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    return False


def pick_weight_dtype(dtype, fallback_dtype, device=None):
    if dtype is None or (dtype_size(dtype) > dtype_size(fallback_dtype)):
        dtype = fallback_dtype
    if not supports_cast(device, dtype):
        dtype = fallback_dtype
    return dtype


def device_supports_non_blocking(device):
    if is_device_mps(device):
        return False
    if is_intel_xpu():
        return False
    if getattr(args, "pytorch_deterministic", False):
        return False
    if directml_enabled:
        return False
    return True


def device_should_use_non_blocking(device):
    if not device_supports_non_blocking(device):
        return False
    return False  # ver comentario original


def force_channels_last():
    return bool(getattr(args, "force_channels_last", False))


def cast_to_device(tensor, device, dtype, copy=False):
    device_supports_cast = False
    if tensor.dtype in (torch.float32, torch.float16):
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if (hasattr(device, 'type') and device.type.startswith("cuda")) or is_intel_xpu():
            device_supports_cast = True

    non_blocking = device_should_use_non_blocking(device)

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
            return tensor.to(device, copy=copy, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
        else:
            return tensor.to(device, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
    else:
        return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)


def xformers_enabled():
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae():
    enabled = xformers_enabled()
    if not enabled:
        return False
    return XFORMERS_ENABLED_VAE


def pytorch_attention_enabled():
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION


def pytorch_attention_flash_attention():
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        if is_nvidia():
            return True
        if is_intel_xpu():
            return True
    return False


def force_upcast_attention_dtype():
    upcast = getattr(args, "force_upcast_attention", False)
    try:
        if platform.mac_ver()[0] in ['14.5']:  # bug negro en OSX Sonoma 14.5
            upcast = True
    except Exception:
        pass
    return torch.float32 if upcast else None


def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            # Estimación conservadora para DirectML
            total = 8 * 1024 * 1024 * 1024
            used = 2 * 1024 * 1024 * 1024  # estimado
            mem_free_total = max(0, total - used)
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats.get('active_bytes.all.current', 0)
            mem_reserved = stats.get('reserved_bytes.all.current', 0)
            mem_free_torch = max(0, mem_reserved - mem_active)
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_total = max(0, mem_free_xpu + mem_free_torch)
        elif torch.cuda.is_available():
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats.get('active_bytes.all.current', 0)
            mem_reserved = stats.get('reserved_bytes.all.current', 0)
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = max(0, mem_reserved - mem_active)
            mem_free_total = max(0, mem_free_cuda + mem_free_torch)
        else:
            mem_free_total = psutil.virtual_memory().available
            mem_free_torch = mem_free_total

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def cpu_mode():
    global cpu_state
    return cpu_state == CPUState.CPU


def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS


def is_device_type(device, type):
    if hasattr(device, 'type'):
        if (device.type == type):
            return True
    return False


def is_device_cpu(device):
    return is_device_type(device, 'cpu')


def is_device_mps(device):
    return is_device_type(device, 'mps')


def is_device_cuda(device):
    return is_device_type(device, 'cuda')


def unet_dtype(device=None, model_params=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    if getattr(args, "unet_in_bf16", False):
        return torch.bfloat16
    if getattr(args, "unet_in_fp16", False):
        return torch.float16
    if getattr(args, "unet_in_fp8_e4m3fn", False):
        return torch.float8_e4m3fn
    if getattr(args, "unet_in_fp8_e5m2", False):
        return torch.float8_e5m2

    for candidate in supported_dtypes:
        if candidate == torch.float16 and should_use_fp16(device, model_params=model_params, prioritize_performance=True, manual_cast=True):
            return candidate
        if candidate == torch.bfloat16 and should_use_bf16(device, model_params=model_params, prioritize_performance=True, manual_cast=True):
            return candidate
    return torch.float32


signal_empty_cache = False


def soft_empty_cache(force=False):
    global cpu_state, signal_empty_cache
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        try:
            if force or is_nvidia():  # en ROCm puede empeorar
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    signal_empty_cache = False
    return


def unload_all_models():
    free_memory(1e30, get_torch_device(), keep_loaded=[], free_all=True)
