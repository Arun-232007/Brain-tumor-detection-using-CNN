import tensorflow as tf

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "Training",
    image_size=(128,128),
    batch_size=32
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "Testing",
    image_size=(128,128),
    batch_size=32
)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

model.save("brain_tumor_model.h5")

print("Model saved successfully!")