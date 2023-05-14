from torchscript.sam_convert import convert_mlir
import argparse

parser = argparse.ArgumentParser(
    description=
    "Export the SAM prompt encoder and mask decoder to an MLIR model.")

parser.add_argument("--checkpoint",
                    type=str,
                    required=True,
                    help="The path to the SAM model checkpoint.")

parser.add_argument("--output",
                    type=str,
                    required=True,
                    help="The filename to save the MLIR model to.")

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help=
    "In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
)

parser.add_argument(
    "--output-type",
    type=str,
    default='raw',
    help=
    "In ['torch', 'tosa', 'linalg-on-tensors', 'stablehlo', 'raw']. The kind of output that `torch_mlir.compile` can produce.",
)

if __name__ == "__main__":
  args = parser.parse_args()
  compiled = convert_mlir(checkpoint=args.checkpoint,
                          model_type=args.model_type,
                          output_type=args.output_type)
  compiled_str = str(compiled)
  print(compiled_str[:500])
  print("...")
  
  print("Writing output to", args.output, "...")
  with open(args.output, "w", encoding="utf-8") as f:
    f.write(compiled_str)
