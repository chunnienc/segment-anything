from torchscript.sam_convert import export_mlir
import argparse

parser = argparse.ArgumentParser(
    description=
    "Export the SAM prompt encoder and mask decoder to an TorchScript model.")

parser.add_argument("--checkpoint",
                    type=str,
                    required=True,
                    help="The path to the SAM model checkpoint.")

parser.add_argument("--output",
                    type=str,
                    required=True,
                    help="The filename to save the TorchScript model to.")

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help=
    "In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
)

if __name__ == "__main__":
  args = parser.parse_args()
  export_mlir(checkpoint=args.checkpoint,
              output=args.output,
              model_type=args.model_type)
