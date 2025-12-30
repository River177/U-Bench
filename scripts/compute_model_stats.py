import argparse
from types import SimpleNamespace
from typing import List, Tuple

import torch


def _patch_torchvision_pretrained_loaders() -> None:
    """Force torchvision model helpers to skip downloading pretrained weights."""

    try:
        import torchvision.models as tvm
    except Exception:
        return

    from functools import wraps
    import inspect

    def make_wrapper(fn):
        if getattr(fn, "_cascade_patched", False):
            return fn

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None

        @wraps(fn)
        def wrapped(*args, **kwargs):
            args_list = list(args)
            if sig is not None:
                params: List[inspect.Parameter] = list(sig.parameters.values())  # type: ignore[name-defined]
                for idx, param in enumerate(params[: len(args_list)]):
                    if param.name == "pretrained":
                        args_list[idx] = False
                    elif param.name == "weights":
                        args_list[idx] = None
            if "pretrained" in kwargs:
                kwargs["pretrained"] = False
            if "weights" in kwargs:
                kwargs["weights"] = None
            kwargs.setdefault("pretrained", False)
            if "weights" not in kwargs:
                kwargs["weights"] = None
            return fn(*tuple(args_list), **kwargs)

        wrapped._cascade_patched = True  # type: ignore[attr-defined]
        return wrapped

    for attr_name in dir(tvm):
        attr = getattr(tvm, attr_name)
        if callable(attr):
            try:
                patched = make_wrapper(attr)
            except Exception:
                continue
            setattr(tvm, attr_name, patched)


def _patch_timm_create_model() -> None:
    """Ensure timm doesn't attempt to load pretrained checkpoints."""
    try:
        import timm
    except Exception:
        return

    orig_create_model = timm.create_model

    def create_model(name, *args, **kwargs):
        kwargs["pretrained"] = False
        return orig_create_model(name, *args, **kwargs)

    timm.create_model = create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Params and FLOPs for multiple models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "MEGANet",
            "EViT_UNet",
            "EMCAD",
            "UTANet",
            "CFFormer",
            "Perspective_Unet",
            "DDANet",
            "CMUNeXt",
            "MobileUViT",
            "RollingUnet",
            "MBSNet",
            "LGMSNet",
            "ResU_KAN",
            "MSLAU_Net",
            "MMUNet",
            "LV_UNet",
            "DDS_UNet",
            "GH_UNet",
            "CSCAUNet",
            "MDSA_UNet",
            "ERDUnet",
            "Tinyunet",
            "CFM_UNet",
            "SimpleUNet",
            "UNetV2",
            "CSWin_UNet",
        ],
    )
    parser.add_argument("--img-size", type=int, default=256, help="Input resolution (square).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic input.")
    parser.add_argument("--input-channel", type=int, default=3, help="Number of input channels.")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of segmentation classes.")
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="Device to run profiling on.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save metrics as JSON.")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.set_grad_enabled(False)

    _patch_torchvision_pretrained_loaders()
    _patch_timm_create_model()

    from thop import clever_format, profile
    from models import build_model

    device = torch.device(args.device)
    dummy = torch.randn(
        args.batch_size,
        args.input_channel,
        args.img_size,
        args.img_size,
        device=device,
    )

    results: List[Tuple[str, str, str, str]] = []
    json_payload = {}

    for model_name in args.models:
        config = SimpleNamespace(model=model_name)
        try:
            model = build_model(config, input_channel=args.input_channel, num_classes=args.num_classes)
            model.eval()
            model.to(device)
            flops, params = profile(model, inputs=(dummy,), verbose=False)
            f_str, p_str = clever_format([flops, params], "%.3f")
            results.append((model_name, p_str, f_str, ""))
            json_payload[model_name] = {"params": p_str, "flops": f_str}
        except Exception as exc:  # pragma: no cover - best effort logging
            err_msg = str(exc)
            results.append((model_name, "-", "-", err_msg))
            json_payload[model_name] = {"error": err_msg}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    header = f"{'Model':<20} {'Params':<12} {'FLOPs':<12} {'Status'}"
    print(header)
    print("-" * len(header))
    for name, params, flops, status in results:
        status_display = status if status else "OK"
        print(f"{name:<20} {params:<12} {flops:<12} {status_display}")

    if args.output_json:
        import json
        from pathlib import Path

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)
        print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
