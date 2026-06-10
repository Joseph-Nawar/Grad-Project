import tensorflow as tf


def build_mobilenetv2_binary_classifier(
    image_size: tuple[int, int] = (160, 160),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """
    Build a lightweight MobileNetV2 transfer-learning binary classifier.

    Backbone:
    - MobileNetV2 pretrained on ImageNet
    - frozen initially

    Head:
    - GlobalAveragePooling2D
    - Dropout
    - Dense sigmoid output
    """
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = False

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mobilenetv2_face_binary_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model

def build_mobilenetv2_multiclass_classifier(
    image_size: tuple[int, int] = (160, 160),
    num_classes: int = 7,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """
    Build a MobileNetV2 transfer-learning multiclass classifier.

    Used for FER2013 facial expression recognition.
    """
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = False

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mobilenetv2_fer2013_multiclass_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    return model