import tensorflow as tf
import pandas as pd

LABEL_MAPPING = {
    "NonStroke": 0,
    "Stroke": 1,
}


def dataframe_to_tf_dataset(
    df: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Create a streaming TensorFlow dataset from image paths.

    This avoids loading the full dataset into RAM.
    """
    paths = df["path"].astype(str).to_list()
    labels = df["class_label"].map(LABEL_MAPPING).astype("int32").to_list()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=seed)

    dataset = dataset.map(
        lambda path, label: load_and_preprocess_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_and_preprocess_image(
    path: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Load image from disk, decode, resize, and normalize.

    Pixel values become 0-1 floats.
    """
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def build_training_augmentation() -> tf.keras.Sequential:
    """
    Mild training-only augmentation.

    We avoid aggressive transforms because facial asymmetry is clinically meaningful.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="face_training_augmentation",
    )


def add_augmentation(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Apply augmentation on the fly to training batches only.
    """
    augmenter = build_training_augmentation()

    def augment_batch(images, labels):
        augmented_images = augmenter(images, training=True)
        augmented_images = tf.clip_by_value(augmented_images, 0.0, 1.0)
        return augmented_images, labels

    return dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)


def dataframe_to_unbatched_tf_dataset(
    df: pd.DataFrame,
    image_size: tuple[int, int],
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Create an unbatched TensorFlow dataset from image paths.

    This is useful for class-balanced sampling where separate class datasets
    are sampled before batching.
    """
    paths = df["path"].astype(str).to_list()
    labels = df["class_label"].map(LABEL_MAPPING).astype("int32").to_list()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: load_and_preprocess_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset


def create_balanced_binary_training_dataset(
    df: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Create a class-balanced training dataset for binary classification.

    It samples 50% from NonStroke and 50% from Stroke using tf.data.
    This avoids materializing augmented copies in memory.
    """
    nonstroke_df = df[df["class_label"] == "NonStroke"].copy()
    stroke_df = df[df["class_label"] == "Stroke"].copy()

    nonstroke_ds = dataframe_to_unbatched_tf_dataset(
        nonstroke_df,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    ).repeat()

    stroke_ds = dataframe_to_unbatched_tf_dataset(
        stroke_df,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    ).repeat()

    balanced_ds = tf.data.Dataset.sample_from_datasets(
        [nonstroke_ds, stroke_ds],
        weights=[0.5, 0.5],
        seed=seed,
    )

    balanced_ds = balanced_ds.batch(batch_size)
    balanced_ds = add_augmentation(balanced_ds)
    balanced_ds = balanced_ds.prefetch(tf.data.AUTOTUNE)

    return balanced_ds