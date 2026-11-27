"""Read-only public-checkpoint feasibility check; synthetic inputs, no training."""
from pathlib import Path
import hashlib
import importlib.util
import json
import torch

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location(
    'verified_public_accnet', ROOT / 'sources/ssl_sslearning_models_accNet.py'
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
out = {'torch_version': torch.__version__, 'training_performed': False,
       'real_dataset_inputs_read': False, 'models': []}
for seconds, filename in [(5, 'ssl_mtl_5_best.mdl'), (10, 'ssl_mtl_best.mdl')]:
    path = ROOT / 'sources' / filename
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = mod.Resnet(output_size=6, is_eva=True, epoch_len=seconds,
                       resnet_version=1)
    expected = model.state_dict()
    normalized = {'.'.join(k.split('.')[1:]): v for k, v in ckpt.items()}
    filtered = {k: v for k, v in normalized.items()
                if k in expected and not k.startswith('classifier.')}
    mismatched = {k: [list(v.shape), list(expected[k].shape)]
                  for k, v in filtered.items() if v.shape != expected[k].shape}
    assert not mismatched, mismatched
    missing_features = [k for k in expected if k.startswith('feature_extractor.')
                        and k not in filtered]
    assert not missing_features, missing_features
    model.load_state_dict(filtered, strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stages = {}
    hooks = [layer.register_forward_hook(
        lambda m, x, y, name=name: stages.update({name: list(y.shape)}))
        for name, layer in model.feature_extractor.named_children()]
    x = torch.linspace(-1, 1, 3 * 30 * seconds).reshape(1, 3, 30 * seconds)
    with torch.inference_mode():
        y = model(x)
    row = {'model': f'harnet{seconds}', 'checkpoint_bytes': path.stat().st_size,
           'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'checkpoint_keys': len(ckpt), 'loaded_feature_state_entries': len(filtered),
           'missing_feature_entries': missing_features,
           'classifier_is_random_untrained': True,
           'parameters_total_six_class_head': sum(p.numel() for p in model.parameters()),
           'parameters_feature_extractor': sum(p.numel() for p in model.feature_extractor.parameters()),
           'input_shape': list(x.shape), 'output_shape': list(y.shape),
           'output_finite': bool(torch.isfinite(y).all()), 'stage_shapes': dict(stages)}
    for hook in hooks:
        hook.remove()
    # UCI 2.56 s at 30 Hz would be about 77 samples, not a 5 or 10 s window.
    try:
        with torch.inference_mode():
            z = model(torch.zeros(1, 3, 77))
        row['77_sample_input'] = {'runs': True, 'output_shape': list(z.shape)}
    except Exception as e:
        row['77_sample_input'] = {'runs': False, 'exception': type(e).__name__,
                                  'message': str(e)}
    out['models'].append(row)
(ROOT / 'harnet_weight_audit.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
