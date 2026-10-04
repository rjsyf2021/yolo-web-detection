# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 rjsyf2021
"""Export the six rectangular engines used by app_ws-multi.py.

Adapted from the project's original export_all_engines.py.
Run on the target GPU server in its existing Ultralytics/TensorRT environment.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile


SIZES = ((1280, 720), (1920, 1080), (640, 480),
         (960, 720), (1280, 960), (1440, 1080))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path('yolo26s.pt'))
    parser.add_argument('--output-dir', type=Path, default=Path('models'))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--sizes', nargs='+', choices=[f'{w}x{h}' for w, h in SIZES],
                        default=[f'{w}x{h}' for w, h in SIZES])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        parser.error('Provide an existing local .pt file with --weights; no automatic download.')
    if not re.fullmatch(r'yolo26[nsmxl]', weights.stem) or weights.suffix != '.pt':
        parser.error('Weight name must be yolo26[nsmxl].pt to match the app scanner.')
    from ultralytics import YOLO
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for size in dict.fromkeys(args.sizes):
        width, height = map(int, size.split('x'))
        destination = args.output_dir.resolve() / f'{weights.stem}_{size}.engine'
        if destination.exists() and not args.overwrite:
            print(f'Skipping existing engine: {destination}', flush=True)
            continue
        # Keep intermediate ONNX/engine files separate from the source weights.
        with tempfile.TemporaryDirectory(prefix='.export-', dir=args.output_dir) as scratch:
            local_weights = Path(scratch) / weights.name
            shutil.copy2(weights, local_weights)
            input_h, input_w = ((height + 31) // 32 * 32, (width + 31) // 32 * 32)
            print(f'Exporting {size}; tensor HxW={input_h}x{input_w}', flush=True)
            exported = Path(YOLO(str(local_weights)).export(
                format='engine', imgsz=(input_h, input_w), half=True,
                device=args.device, batch=1, dynamic=False))
            if not exported.is_file() or not exported.stat().st_size:
                raise RuntimeError(f'Exporter did not produce a nonempty engine: {exported}')
            exported.replace(destination)
        metadata = dict(weights=weights.name, nominal_width=width, nominal_height=height,
                        input_width=input_w, input_height=input_h, half=True,
                        batch=1, dynamic=False, device=args.device)
        destination.with_suffix('.json').write_text(
            json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        print(f'Saved: {destination}', flush=True)


if __name__ == '__main__':
    main()
