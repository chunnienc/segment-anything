import os
import torch
import torch_mlir
from segment_anything import sam_model_registry
from .patches import apply_patches
from .sam_predictor_base_model import SamPredictorBaseModel, TupleOutputSamPredictorBaseModel


def _get_model(model_cls, checkpoint: str, model_type: str):
  apply_patches()
  # An instance of the model.
  base_model = sam_model_registry[model_type](checkpoint=checkpoint)

  model = model_cls(model=base_model)
  model.eval()

  return model


def _get_example_input():
  # An example input you would normally provide to your model's forward() method.
  B = 1
  N = 1
  H = 1024
  W = 1024
  example_input = torch.randint(0, 255, size=(3, H, W))
  return example_input


def convert_torchscript(checkpoint: str, model_type: str):
  model = _get_model(SamPredictorBaseModel, checkpoint, model_type)
  example_input = _get_example_input()

  # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
  # This is commented as it does work work with the SAM model.
  # traced_script_module = torch.jit.trace(naive_model, example_inputs)

  scripted_model = torch.jit.script(model,
                                    example_inputs={model: [(example_input,)]})
  # This is also commented as it does work work with the SAM model.
  # script_module = torch.jit.optimize_for_inference(script_module)

  # Preview the TorchScript model
  print(scripted_model(example_input))

  return scripted_model


def convert_mlir(checkpoint: str, model_type: str, output_type):
  model = _get_model(TupleOutputSamPredictorBaseModel, checkpoint, model_type)
  example_input = _get_example_input()

  return torch_mlir.compile(
      model,
      [example_input],
      use_tracing=False,
      output_type=output_type,
      verbose=True,
  )
