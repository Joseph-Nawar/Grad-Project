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


def build_broad_constrained_image_automodel(
    max_trials: int = 3,
    project_name: str = "face_autokeras_broad_160",
    directory: str | None = None,
    overwrite: bool = True,
) -> ak.AutoModel:
    """
    Build a broader AutoKeras image model under project constraints.

    Difference from Trial 001:
    - block_type is left as None, so AutoKeras can tune the image block type.
    - normalize is False because the tf.data pipeline already normalizes images.
    - augment is False because augmentation is applied externally only to training data.

    This gives AutoKeras more freedom than the vanilla-only search, while
    preserving control over memory and data handling.
    """
    input_node = ak.ImageInput()

    output_node = ak.ImageBlock(
        block_type=None,
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