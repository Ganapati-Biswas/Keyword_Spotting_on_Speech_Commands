import argparse
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import numpy as np
import os

class RandomDataReader(CalibrationDataReader):
    def __init__(self, input_name, shape, num_samples=100):
        self.input_name = input_name
        self.data = [np.random.randn(*shape).astype(np.float32) for _ in range(num_samples)]
        self.enum_data = iter(self.data)
        
    def get_next(self):
        nxt = next(self.enum_data, None)
        if nxt is not None:
            return {self.input_name: nxt}
        return None

def quantize(args):
    # Assume input is log mel
    dr = RandomDataReader('log_mel', (1, 1, 40, 98))
    
    quantize_static(
        model_input=args.input_model,
        model_output=args.output_model,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        optimize_model=True
    )
    
    print(f"Quantized model saved to {args.output_model}")
    print(f"Original size: {os.path.getsize(args.input_model)/1024:.2f} KB")
    print(f"Quantized size: {os.path.getsize(args.output_model)/1024:.2f} KB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', type=str, required=True)
    parser.add_argument('--output_model', type=str, default='model_quant.onnx')
    args = parser.parse_args()
    quantize(args)
