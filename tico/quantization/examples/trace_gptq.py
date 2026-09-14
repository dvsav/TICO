#!/usr/bin/env python3
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.debug.trace import trace_ptq_parity
from tico.quantization.recipes.adapters import get_adapter
from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.context import RecipeContext

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Trace and evaluate PEIR for GPTQ-quantized models."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to recipe YAML/JSON config for quantization.",
    )

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values using dotted paths.",
    )

    return parser.parse_args()


def print_header(header: str, char: str = "*") -> None:
    """Print a formatted header with centered text."""
    print()
    print(char * 80)
    print(f"{char} {header :^76} {char}")
    print(char * 80)
    print()


def main() -> None:
    """Main entry point for the trace_gptq script."""
    args = parse_arguments()

    # Load config with overrides
    overrides = list(args.set)
    cfg = load_recipe_config(args.config, overrides=overrides)

    # Disable evaluation and export stages (we only want quantization)
    if "evaluation" in cfg:
        cfg["evaluation"]["enabled"] = False
    if "export" in cfg:
        cfg["export"]["enabled"] = False

    adapter = get_adapter(cfg["model"]["family"])
    ctx = RecipeContext(cfg=cfg, adapter=adapter)
    ctx = adapter.load_model(ctx)
    ctx.calibration_inputs = adapter.build_calibration_inputs(ctx)

    trace_ptq_parity(
        ctx,
        enable_quantization=args.enable_quantization,
        interesting_modules=args.interesting_modules,
    )


if __name__ == "__main__":
    main()
