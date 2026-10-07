import argparse
import torch
import onnx
from models import get_model
from features import LogMelFeatureExtractor
import torch.nn as nn

class ModelWithFrontend(nn.Module):
    def __init__(self, model, frontend):
        super().__init__()
        self.frontend = frontend
        self.model = model
        
    def forward(self, x):
        return self.model(self.frontend(x))

def export_onnx(args):
    model = get_model(args.model_name, num_classes=12)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()
    
    if args.include_frontend:
        frontend = LogMelFeatureExtractor()
        model = ModelWithFrontend(model, frontend)
        dummy_input = torch.randn(1, 16000)
        input_names = ['waveform']
    else:
        dummy_input = torch.randn(1, 1, 40, 98)
        input_names = ['log_mel']
        
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=['logits'],
        dynamic_axes={input_names[0]: {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    )
    
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print(f"Successfully exported to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='model.onnx')
    parser.add_argument('--include_frontend', action='store_true')
    args = parser.parse_args()
    export_onnx(args)
