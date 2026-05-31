import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score

SEED = 42
RESULTS_DIR = Path("resultados")
FEATURE_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
TARGET_COL = "class"


def configurar_entorno():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=str(RESULTS_DIR / "corrida.log"),
        filemode="w",
        encoding="utf-8",
    )
    return logging.getLogger(__name__)


def cargar_datos(logger):
    df = pd.read_csv("star_classification.csv")
    logger.info("Dimensiones del dataset: %s", df.shape)
    logger.info("Distribución de clases:\n%s\n", df["class"].value_counts())

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    logger.info("Features usadas (%d): %s", len(FEATURE_COLS), FEATURE_COLS)
    return X, y


def codificar_labels(y, logger):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_
    logger.info("Clases: %s\n", class_names)
    return y_enc, class_names


def dividir_dataset(X, y_enc, logger):
    # 60% entrenamiento / 20% val / 20% test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=SEED, stratify=y_enc
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=SEED, stratify=y_tv  # 0.25 * 0.80 = 0.20
    )
    logger.info(
        "Train: %s  |  Val: %s  |  Test: %s",
        f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}",
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalizar(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test


def one_hot(y, n_classes):
    return tf.keras.utils.to_categorical(y, n_classes)

def graficar_loss(history, titulo, filename, logger):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="Entrenamiento")
    ax1.plot(history.history["val_loss"], label="Validación")
    ax1.set_title(f"{titulo} — Pérdida")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Pérdida")
    ax1.legend(); ax1.grid(True)

    ax2.plot(history.history["accuracy"], label="Entrenamiento")
    ax2.plot(history.history["val_accuracy"], label="Validación")
    ax2.set_title(f"{titulo} — Precisión")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Precisión")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    ruta = RESULTS_DIR / filename
    plt.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info("  → Guardado: %s", ruta)


def evaluar_modelo(model, X_data, y_true, split_name, titulo, class_names, logger):
    y_pred = np.argmax(model.predict(X_data, verbose=0), axis=1)
    n_classes = len(class_names)

    cm = confusion_matrix(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, labels=range(n_classes))
    rec = recall_score(y_true, y_pred, average=None, labels=range(n_classes))

    logger.info("\n  [%s] Matriz de confusión:", split_name)
    logger.info("%s", pd.DataFrame(cm, index=class_names, columns=class_names).to_string())

    logger.info("\n  [%s] Métricas por clase:", split_name)
    for i, cls in enumerate(class_names):
        logger.info("    %s  Precisión=%.4f  Recall=%.4f", cls, prec[i], rec[i])

    # heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{titulo} — {split_name}")
    ax.set_ylabel("Real"); ax.set_xlabel("Predicho")
    plt.tight_layout()
    fname = RESULTS_DIR / f"cm_{titulo.replace(' ', '_')}_{split_name}.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info("  → Guardado: %s", fname)

    return y_pred

def generar_mlp_superficial(input_dim, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="Shallow_MLP")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def generar_mlp_profundo(input_dim, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="Deep_MLP")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def generar_mlp_regularizado(input_dim, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="Regularized_Deep_MLP")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def generar_cnn(input_dim, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((input_dim, 1)),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="CNN_1D")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def entrenar_y_evaluar(configs, X_train, X_val, X_test,
                       y_train, y_val, y_test,
                       y_train_oh, y_val_oh, y_test_oh,
                       class_names, logger):
    resultados = []

    for titulo, build_fn, epochs in configs:
        logger.info("\n" + "=" * 60)
        logger.info("  %s", titulo)
        logger.info("=" * 60)

        model = build_fn()
        model.summary(print_fn=logger.info)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=epochs,
            batch_size=512,
            callbacks=[early_stop],
            verbose=1,
        )

        tag = titulo.split("—")[0].strip().replace(" ", "_")
        graficar_loss(history, titulo, f"loss_{tag}.png", logger)

        logger.info("\n  -- Evaluación entrenamiento --")
        evaluar_modelo(model, X_train, y_train, "Entrenamiento", titulo, class_names, logger)

        logger.info("\n  -- Evaluación prueba --")
        evaluar_modelo(model, X_test, y_test, "Prueba", titulo, class_names, logger)

        test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
        logger.info("\n  Precisión final en prueba: %.4f  |  pérdida: %.4f", test_acc, test_loss)

        resultados.append({
            "Modelo": titulo,
            "Precisión prueba": test_acc,
            "Pérdida prueba": test_loss,
            "Mejor precisión entrenamiento": max(history.history["accuracy"]),
            "Mejor precisión validación": max(history.history["val_accuracy"]),
            "Mejor pérdida entrenamiento": min(history.history["loss"]),
            "Mejor pérdida validación": min(history.history["val_loss"]),
        })

    return resultados


def guardar_resumen(resultados, logger):
    logger.info("\n" + "=" * 60)
    logger.info("  RESUMEN COMPARATIVO")
    logger.info("=" * 60)

    summary_df = pd.DataFrame(resultados)
    logger.info("\n%s", summary_df.to_string(index=False))

    # grafico de barras comparando accuracy en test
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [r["Modelo"].split("—")[1].strip() for r in resultados]
    accs = [r["Precisión prueba"] for r in resultados]

    bars = ax.bar(labels, accs, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.bar_label(bars, fmt="%.4f", padding=3)

    ax.set_ylim(max(0.8, min(accs) - 0.05), 1.01)
    ax.set_ylabel("Precisión en prueba")
    ax.set_title("Comparación de modelos — Precisión en prueba")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("→ Guardado: %s", RESULTS_DIR / "model_comparison.png")

    with open(RESULTS_DIR / "metricas_finales.txt", "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(f"{r['Modelo']}\n")
            f.write(f"MEJOR PRECISION EN EL ENTRENAMIENTO: {r['Mejor precisión entrenamiento']:.4f}\n")
            f.write(f"MEJOR PRECISION EN VALIDACION: {r['Mejor precisión validación']:.4f}\n")
            f.write(f"MEJOR PERDIDA EN EL ENTRENAMIENTO: {r['Mejor pérdida entrenamiento']:.4f}\n")
            f.write(f"MEJOR PERDIDA EN VALIDACION: {r['Mejor pérdida validación']:.4f}\n\n")
    logger.info("→ Guardado: %s", RESULTS_DIR / "metricas_finales.txt")

def main():
    logger = configurar_entorno()

    X, y = cargar_datos(logger)
    y_enc, class_names = codificar_labels(y, logger)
    n_classes = len(class_names)

    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dataset(X, y_enc, logger)
    X_train, X_val, X_test = normalizar(X_train, X_val, X_test)

    y_train_oh = one_hot(y_train, n_classes)
    y_val_oh = one_hot(y_val, n_classes)
    y_test_oh = one_hot(y_test, n_classes)

    input_dim = X_train.shape[1]

    configs = [
        ("Entrenamiento 1 — MLP Superficial",  lambda: generar_mlp_superficial(input_dim, n_classes),  250),
        ("Entrenamiento 2 — MLP Profundo",     lambda: generar_mlp_profundo(input_dim, n_classes),     250),
        ("Entrenamiento 3 — MLP Regularizado", lambda: generar_mlp_regularizado(input_dim, n_classes), 250),
        ("Entrenamiento 4 — CNN 1D",           lambda: generar_cnn(input_dim, n_classes),       250),
    ]

    resultados = entrenar_y_evaluar(
        configs, X_train, X_val, X_test,
        y_train, y_val, y_test,
        y_train_oh, y_val_oh, y_test_oh,
        class_names, logger,
    )

    guardar_resumen(resultados, logger)


if __name__ == "__main__":
    main()