"""MLCube handler file"""
import typer
import yaml
import os
from infer import run_inference

app = typer.Typer()


@app.command("infer")
def infer(
    data_path: str = typer.Option(..., "--data_path"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_path: str = typer.Option(..., "--output_path"),
    # Provide additional parameters as described in the mlcube.yaml file
    # e.g. model weights:
    ckpts_path: str = typer.Option(..., "--ckpts_path")
):
    with open(parameters_file) as f:
        parameters = yaml.safe_load(f)

    parameters["prep_data"] = data_path
    parameters["prep_results"] = data_path
    parameters["data"] = os.path.join(data_path, "12_3d")
    parameters["results"] = os.path.join(output_path, "results")

    run_inference(data_path, parameters, output_path, ckpts_path)


@app.command("hotfix")
def hotfix():
    # NOOP command for typer to behave correctly. DO NOT REMOVE OR MODIFY
    pass


if __name__ == "__main__":
    app()
