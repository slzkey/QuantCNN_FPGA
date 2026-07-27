# QuantCNN-FPGA

INT8-quantized CNN inference on a Xilinx Artix-7, using a MAC datapath written in
SystemVerilog. Covers the full path: PyTorch training → INT8 quantization → integer
reference model → RTL → on-board inference.

INT8 量化 CNN 在 Xilinx Artix-7 上的推理实现，MAC 数据通路用 SystemVerilog 手写。
覆盖完整链路：PyTorch 训练 → INT8 量化 → 整数参考模型 → RTL → 板上推理。

**Status:** work in progress — Stage 0 / 5

```
PyTorch (FP32) → INT8 → integer reference model → SystemVerilog → Artix-7
```

## Spec

| | |
|---|---|
| Task | MNIST, 10-class |
| Network | Conv(1→8,3×3)–ReLU–Pool – Conv(8→16,3×3)–ReLU–Pool – FC(784→10) |
| Parameters | ≈9k, ≈9 KB at INT8 (fits on-chip BRAM) |
| Quantization | symmetric per-tensor INT8, 32-bit accumulator, shift-based rescale |
| Board | Digilent Nexys Video (Artix-7 XC7A200T) |
| Tools | PyTorch, Brevitas, Vivado, Questa |

## Results

Pending. 待补。

| Config | Accuracy | Latency | LUT | FF | DSP | BRAM | Fmax |
|---|---|---|---|---|---|---|---|
| FP32 (ref) | — | — | — | — | — | — | — |
| INT8 | — | — | — | — | — | — | — |
| INT4 | — | — | — | — | — | — | — |

## Usage

```bash
# train FP32 baseline
python python/train_fp32.py

# quantize to INT8, export weights and test vectors
python python/quantize.py
python python/export_weights.py

# integer reference model (bit-accurate, used as simulation golden)
python python/golden_int.py

# RTL simulation
cd tb && make sim
```

## Layout

```
python/   training, quantization, integer reference model, weight export
rtl/      MAC array, conv/FC engines, rescale, pooling, top
tb/       testbench driven by reference-model vectors
vivado/   constraints, build scripts
results/  accuracy, utilization, timing reports
```

## Notes

- The integer reference model uses no floating point and is bit-exact with the RTL;
  it is the golden model for simulation.
- Weights and test images are preloaded into BRAM — no external memory, no UART path.
- FC engine is implemented first, then the convolution engine.

## License

MIT
