import autokeras as ak


def build_constrained_image_automodel(
    block_type: str = "vanilla",
    max_trials: int = 2,
    project_name: str = "face_autokeras_constrained",
    directory: str | None = None,
    overwrite: bool = True,
) -> ak.AutoModel:
    """
    Build a constrained AutoKeras image model.

    block_type options:
    - "vanilla": smaller CNN-style search space
    - "resnet": stronger but heavier
    - "xception": usually heavier

    For rural/offline deployment experiments, start with "vanilla".
    """
    input_node = ak.ImageInput()

    output_node = ak.ImageBlock(
        block_type=block_type,
        normalize=False,
        augment=False,
    )(input_node)

    output_node = ak.ClassificationHead(num_classes=2)(output_node)

    return ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        max_trials=max_trials,
        overwrite=overwrite,
        project_name=project_name,
        directory=directory,
    )