"""
Functions needed to load parameters from params.yaml tracked with DVC
"""
import sys
import os

import yaml


def get_params(stage_fn: str = None):
    """
    Reads parameters for a given DVC stage from params.yaml.

    The stage name is inferred from the name of the python file that calls this
    function.
    Args:
        stage_fn (str): Name of the stage. If None, the name of the file
            that calls this function is used. Defaults to None.
    Returns:
        dict with parameters for the stage
    Raises:
        KeyError: if the stage name is not found in params.yaml
    """

    if stage_fn is None:
        stage_fn = os.path.basename(sys.argv[0]).replace(".py", "")

    try:
        params = yaml.safe_load(open("params.yaml"))[stage_fn]
    except KeyError as exc:
        print(f'ERROR: Key "{stage_fn}" not in parameters.yaml.')
        raise KeyError(f"Is the stage file name ({sys.argv[0]}) " +
                       "the same as the stage name in params.yaml?") from exc
    try:
        all_params = yaml.safe_load(open("params.yaml"))['all']
        params = {**params, **all_params}
    except KeyError:
        print(
            'WARNING: Key "all" not in parameters.yaml.' +
            'Only returning stage parameters.')

    return params


def get_stage_inputs_outputs_metrics(params: dict = None):
    """
    Get the inputs, outputs and metrics files paths for the stage from
      params.yaml.

    If the params dict is not provided, the params.yaml file is read.

    The function allows for a stage input/output to be specified as the
    input/output of another stage. This is done by using the & character
    followed by the stage name and the parameter name. For example, if the
    input_path is &stage_name.parameter_name, the function will look for
    the parameter parameter_name in the stage stage_name and use its value
    as the input_path.

    Args:
        params (dict, optional): Dictionary with the params of that
        stage. Defaults to None.

    Returns:
        dict: Dictionary with the paths to the inputs, outputs and metrics
    """

    if params is None:
        params = get_params()

    try:
        inputs = params['input_path']
        # Check if the input path is a dictionary
        if isinstance(inputs, dict):
            for key, value in inputs.items():
                if value[0] == '&':
                    stage = value[1:].split('.')[0]
                    param = value[1:].split('.')[1]
                    inputs[key] = get_params(stage)[param]
        else:
            if inputs[0] == '&':
                stage = inputs[1:].split('.')[0]
                param = inputs[1:].split('.')[1]
                inputs = get_params(stage)[param]
    except KeyError as exc:
        print(
            'ERROR: Key "input_path" not in'
            f' parameters.yaml for stage {params}.')
        raise KeyError("Is the stage name correct?") from exc

    try:
        outputs = params['output_path']
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if value[0] == '&':
                    stage = value[1:].split('.')[0]
                    param = value[1:].split('.')[1]
                    outputs[key] = get_params(stage)[param]
        else:
            if outputs[0] == '&':
                stage = outputs[1:].split('.')[0]
                param = outputs[1:].split('.')[1]
                outputs = get_params(stage)[param]
    except KeyError as exc:
        print(
            'ERROR: Key "output_path" not'
            f' in parameters.yaml for stage {params}.')
        raise KeyError("Is the stage name correct?") from exc
    if 'metrics_path' in params:
        metrics = params['metrics_path']
    else:
        metrics = None

    return {
        "input_path": inputs,
        "output_path": outputs,
        "metrics_path": metrics
    }
