import argparse
import glob
import os
import sys
import time
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
from features import LogMelFeatureExtractor
from models import get_model

LABELS = [
    'yes', 'no', 'up', 'down', 'left', 'right',
    'on', 'off', 'stop', 'go', 'unknown', 'silence'
]
TARGET_WORDS = set(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'])
SAMPLE_RATE = 16000


def find_latest_checkpoint():
    candidates = glob.glob('runs/**/best_model.pth', recursive=True)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def sync_device(device: torch.device):
    """Ensure GPU/MPS operations complete for accurate latency benchmarking."""
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def get_model_computation_profile(model: nn.Module, input_shape=(1, 1, 40, 98), device='cpu'):
    """Compute total parameters, trainable parameters, and MACs/FLOPs."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_macs = 0
    hooks = []

    def hook(m, x, y):
        nonlocal total_macs
        if isinstance(m, nn.Conv2d):
            b, c_out, h_out, w_out = y.shape
            macs = b * c_out * h_out * w_out * (m.in_channels * m.kernel_size[0] * m.kernel_size[1]) // m.groups
            total_macs += macs
        elif isinstance(m, nn.Linear):
            total_macs += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))

    with torch.no_grad():
        dummy_x = torch.zeros(input_shape, device=device)
        model(dummy_x)

    for h in hooks:
        h.remove()

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'macs': total_macs,
        'flops': total_macs * 2,
        'param_size_kb': (total_params * 4) / 1024.0  # FP32 bytes
    }


def load_kws_model(model_name: str, checkpoint_path: str, device: torch.device):
    model = get_model(model_name, num_classes=len(LABELS)).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    frontend = LogMelFeatureExtractor().to(device)
    frontend.eval()

    comp_profile = get_model_computation_profile(model, device=device)
    return model, frontend, comp_profile


def predict_with_metrics(waveform: np.ndarray, model, frontend, device: torch.device, comp_profile: dict):
    # Normalize amplitude to match standard SpeechCommands peak amplitude
    peak = np.abs(waveform).max()
    if peak > 0.01:
        waveform = (waveform / peak) * 0.65

    tensor = torch.from_numpy(waveform.copy()).float().unsqueeze(0).to(device)  # (1, 16000)

    sync_device(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        features = frontend(tensor)
        sync_device(device)
        t1 = time.perf_counter()

        outputs = model(features)
        sync_device(device)
        t2 = time.perf_counter()

        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    fe_time_ms = (t1 - t0) * 1000.0
    model_time_ms = (t2 - t1) * 1000.0
    total_time_ms = (t2 - t0) * 1000.0

    # Device memory profiling
    mem_mb = 0.0
    if device.type == 'mps':
        mem_mb = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
    elif device.type == 'cuda':
        mem_mb = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)

    metrics = {
        'fe_time_ms': fe_time_ms,
        'model_time_ms': model_time_ms,
        'total_time_ms': total_time_ms,
        'rtf': total_time_ms / 1000.0,
        'throughput_fps': 1000.0 / max(total_time_ms, 1e-4),
        'macs': comp_profile['macs'],
        'flops': comp_profile['flops'],
        'params': comp_profile['total_params'],
        'param_size_kb': comp_profile['param_size_kb'],
        'mem_mb': mem_mb,
        'peak_amplitude': float(peak)
    }

    return probs, metrics


def run_record_mode(model, frontend, device, comp_profile, threshold):
    print("\n" + "=" * 65)
    print("MODE: Push-to-Talk (Record 1-second clips with Profiling)")
    print(f"Target Keywords : {', '.join(sorted(list(TARGET_WORDS)))}")
    print(f"Model Complexity: {comp_profile['macs']/1e6:.2f} MMACs | {comp_profile['total_params']:,} params ({comp_profile['param_size_kb']:.1f} KB)")
    print("=" * 65)

    while True:
        try:
            input("\n👉 Press [ENTER] to record 1 second (or Ctrl+C to quit)...")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        print("🎙️  Recording (1.0s)... Speak now!", end='', flush=True)
        audio = sd.rec(int(SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        print(" Done!")

        waveform = audio.flatten()
        probs, metrics = predict_with_metrics(waveform, model, frontend, device, comp_profile)

        # Sort predictions
        top_indices = np.argsort(probs)[::-1][:3]
        top_label = LABELS[top_indices[0]]
        top_prob = probs[top_indices[0]]

        if top_prob >= threshold and top_label in TARGET_WORDS:
            print(f"\n🎯 DETECTED KEYWORD: \033[1;32m{top_label.upper()}\033[0m ({top_prob*100:.1f}%)")
        else:
            print(f"\nResult: {top_label} ({top_prob*100:.1f}%)")

        print("Top 3 Predictions:")
        for idx in top_indices:
            label = LABELS[idx]
            p = probs[idx]
            bar = "█" * int(p * 30)
            print(f"  {label:<10} [{bar:<30}] {p*100:5.1f}%")

        # Detailed Profiling Output
        print("\n⚡ Inference & Computation Profiling:")
        print(f"  • Latency Breakdown  : Total: {metrics['total_time_ms']:.2f} ms (Audio FE: {metrics['fe_time_ms']:.2f} ms | Model: {metrics['model_time_ms']:.2f} ms)")
        print(f"  • Real-Time Factor   : {metrics['rtf']:.4f}x ({1.0/max(metrics['rtf'], 1e-6):.0f}x faster than real-time)")
        print(f"  • Throughput         : ~{metrics['throughput_fps']:.0f} inferences/second")
        print(f"  • Computational Cost : {metrics['macs']/1e6:.2f} MMACs (~{metrics['flops']/1e6:.2f} MFLOPs per 1s clip)")
        print(f"  • Model Parameters   : {metrics['params']:,} (FP32 weight size: ~{metrics['param_size_kb']:.1f} KB)")
        print(f"  • Hardware Backend   : {device} ({'Apple Silicon MPS' if device.type == 'mps' else device.type.upper()})")
        if metrics['mem_mb'] > 0:
            print(f"  • Device Memory      : {metrics['mem_mb']:.2f} MB allocated")


def run_stream_mode(model, frontend, device, comp_profile, threshold, step_ms=150):
    print("\n" + "=" * 65)
    print("MODE: Real-Time Continuous Streaming with Profiling")
    print(f"Target Keywords     : {', '.join(sorted(list(TARGET_WORDS)))}")
    print(f"Confidence Threshold: {threshold*100:.0f}%")
    print(f"Model Complexity    : {comp_profile['macs']/1e6:.2f} MMACs | {comp_profile['total_params']:,} params")
    print("Press Ctrl+C to stop.")
    print("=" * 65 + "\n")

    window_size = SAMPLE_RATE  # 1.0 second buffer
    step_size = int(SAMPLE_RATE * (step_ms / 1000.0))  # update every step_ms
    buffer = np.zeros(window_size, dtype=np.float32)

    debounce_seconds = 1.0
    last_detect_time = 0.0
    last_detected_word = ""

    def audio_callback(indata, frames, time_info, status):
        nonlocal buffer
        chunk = indata[:, 0]
        buffer = np.roll(buffer, -len(chunk))
        buffer[-len(chunk):] = chunk

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=step_size,
        dtype='float32',
        callback=audio_callback
    )

    with stream:
        print("🎙️  Listening to microphone in real time...\n")
        try:
            while True:
                time.sleep(step_ms / 1000.0)
                current_time = time.time()

                # Calculate RMS volume level
                rms = np.sqrt(np.mean(buffer[-step_size:] ** 2))
                level = min(int(rms * 100), 15)
                vu_meter = "🔉 " + "■" * level + " " * (15 - level)

                # Skip inference if near silence to save compute
                if rms < 0.005:
                    sys.stdout.write(f"\r{vu_meter} | [Listening...]                                              ")
                    sys.stdout.flush()
                    continue

                probs, metrics = predict_with_metrics(buffer, model, frontend, device, comp_profile)
                top_idx = int(np.argmax(probs))
                top_label = LABELS[top_idx]
                top_prob = float(probs[top_idx])

                if top_label in TARGET_WORDS and top_prob >= threshold:
                    # Debounce duplicate triggers
                    if (current_time - last_detect_time > debounce_seconds) or (top_label != last_detected_word):
                        last_detect_time = current_time
                        last_detected_word = top_label
                        sys.stdout.write("\r" + " " * 80 + "\r")
                        print(
                            f"🎯 [{time.strftime('%H:%M:%S')}] DETECTED: \033[1;32m{top_label.upper():<6}\033[0m "
                            f"({top_prob*100:5.1f}%) | "
                            f"Latency: {metrics['total_time_ms']:4.2f}ms (FE: {metrics['fe_time_ms']:.2f}ms, Model: {metrics['model_time_ms']:.2f}ms) | "
                            f"RTF: {metrics['rtf']:.4f}x | {metrics['macs']/1e6:.1f}M MACs"
                        )
                else:
                    sys.stdout.write(
                        f"\r{vu_meter} | Top: {top_label:<8} ({top_prob*100:4.1f}%) | "
                        f"Inf: {metrics['total_time_ms']:4.2f}ms ({metrics['throughput_fps']:.0f} FPS) | "
                        f"{metrics['macs']/1e6:.1f}M MACs  "
                    )
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\nStopped listening.")


def main():
    parser = argparse.ArgumentParser(description="Live Audio Keyword Spotting Inference with Profiling")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to best_model.pth")
    parser.add_argument('--model_name', type=str, default='baselinecnn', help="Model architecture")
    parser.add_argument('--mode', type=str, choices=['stream', 'record'], default='stream',
                        help="'stream' for continuous real-time listening, 'record' for 1-second clips")
    parser.add_argument('--threshold', type=float, default=0.50, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument('--device', type=str, default='auto', help="mps, cpu, or auto")
    args = parser.parse_args()

    # Determine checkpoint
    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("❌ Error: Could not find model checkpoint. Train a model first or pass --checkpoint <path>.")
        sys.exit(1)

    print(f"📦 Loading checkpoint: {checkpoint_path}")

    # Determine device
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"⚡ Device: {device} ({'Apple Silicon MPS' if device.type == 'mps' else device.type.upper()})")

    model, frontend, comp_profile = load_kws_model(args.model_name, checkpoint_path, device)
    print("✅ Model loaded successfully!")
    print(f"📊 Architecture Stats: {comp_profile['total_params']:,} params (~{comp_profile['param_size_kb']:.1f} KB) | {comp_profile['macs']/1e6:.2f} MMACs (~{comp_profile['flops']/1e6:.2f} MFLOPs)")

    if args.mode == 'record':
        run_record_mode(model, frontend, device, comp_profile, args.threshold)
    else:
        run_stream_mode(model, frontend, device, comp_profile, args.threshold)


if __name__ == '__main__':
    main()
