import tensorflow as tf


def build_face_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Build safe image augmentation for face classification.

    These augmentations are intentionally mild because facial asymmetry
    is clinically relevant and should not be distorted heavily.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
            tf.keras.layers.RandomBrightness(0.10),
        ],
        name="face_training_augmentation",
    )


def apply_augmentation_to_training_data(
    X_train,
    y_train,
    target_minority_ratio: float = 0.50,
    minority_class: int = 1,
    seed: int = 42,
):
    """
    Deprecated for main training pipeline.

    This function materializes augmented images in memory and may cause memory
    pressure for larger datasets. Prefer tf.data on-the-fly augmentation in
    preprocessing/tf_image_dataset.py.
    """

    """
    Create additional augmented samples for the minority class only.

    This returns expanded arrays for training only.

    Validation and test data must never be augmented.
    """
    tf.random.set_seed(seed)

    class_counts = {
        int(label): int((y_train == label).sum())
        for label in set(y_train.tolist())
    }

    majority_count = max(class_counts.values())
    minority_count = class_counts[minority_class]

    target_minority_count = int(majority_count * target_minority_ratio / (1 - target_minority_ratio))
    needed = max(0, target_minority_count - minority_count)

    if needed == 0:
        return X_train, y_train

    minority_indices = [i for i, label in enumerate(y_train) if label == minority_class]
    augmentation_model = build_face_augmentation_pipeline()

    augmented_images = []
    augmented_labels = []

    for i in range(needed):
        source_index = minority_indices[i % len(minority_indices)]
        image = X_train[source_index]

        image_batch = tf.expand_dims(image, axis=0)
        augmented = augmentation_model(image_batch, training=True)[0].numpy()

        augmented_images.append(augmented)
        augmented_labels.append(minority_class)

    X_augmented = tf.concat([X_train, tf.convert_to_tensor(augmented_images)], axis=0).numpy()
    y_augmented = tf.concat([y_train, tf.convert_to_tensor(augmented_labels, dtype=y_train.dtype)], axis=0).numpy()

    return X_augmented, y_augmented