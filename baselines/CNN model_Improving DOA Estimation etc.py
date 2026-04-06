#CNN model described in the paper: Improving DOA Estimation via an Optimal Deep Residual Neural Network Classifier on Uniform Linear Arrays

import os, random
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

d = scipy.io.loadmat("") #path here
y_receive = d["y_receive"][:, :, 2, :]              # (16,100,5000) sized dataset at 30 dB SNR
target_azimuth = np.squeeze(d["target_azimuth"]).T # (5000,4) radians

N = y_receive.shape[-1]  # 5000
M = y_receive.shape[0]   # 16
snapshots = y_receive.shape[1]  # 100

cov = np.zeros((M, M, N), dtype=np.complex64)
for i in range(N):
    y = np.asarray(y_receive[:, :, i])                       # (16,100)
    R = (y @ y.conj().T) / snapshots                          # (16,16)
    R = R / (np.trace(R).real + 1e-8)
    cov[:, :, i] = R

indices = np.round(target_azimuth * 180.0 / np.pi + 90.0).astype(np.int64)
indices = np.clip(indices, 0, 180)  # (N,4)
one_hot_target_azimuth = np.zeros((N, 4, 181), dtype=np.float32)
i_idx = np.arange(N)[:, None]
j_idx = np.arange(4)[None, :]
one_hot_target_azimuth[i_idx, j_idx, indices] = 1.0

cov_2ch = np.stack([cov.real, cov.imag], axis=-1)     # (16,16,N,2)
X = np.transpose(cov_2ch, (2, 0, 1, 3)).astype(np.float32)  # (N,16,16,2)
Y = one_hot_target_azimuth.astype(np.float32)               # (N,4,181)

perm = np.random.permutation(N)
ntr = int(0.8 * N)
tr_idx, va_idx = perm[:ntr], perm[ntr:]
X_tr, Y_tr = X[tr_idx], Y[tr_idx]
X_va, Y_va = X[va_idx], Y[va_idx]

#print("X_tr:", X_tr.shape)
#print("Y_tr:", Y_tr.shape)

def build_model(num_classes=181, num_targets=4):
    inp = keras.Input(shape=(16, 16, 2), name="cov_2ch")  

    # Block 1
    x = layers.Conv2D(512, (3, 3), padding="same", use_bias=False, name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    
    x = layers.ZeroPadding2D(padding=1, name="pad1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid", name="pool1")(x)
    x = layers.BatchNormalization(name="bn1_postpool")(x)
    x = layers.ReLU(name="relu1_postpool")(x)

    # Block 2
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.ZeroPadding2D(padding=1, name="pad2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid", name="pool2")(x)
    x = layers.BatchNormalization(name="bn2_postpool")(x)
    x = layers.ReLU(name="relu2_postpool")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.ZeroPadding2D(padding=1, name="pad3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid", name="pool3")(x)
    x = layers.BatchNormalization(name="bn3_postpool")(x)
    x = layers.ReLU(name="relu3_postpool")(x)

    # Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    x = layers.Dense(num_targets * num_classes, name="fc")(x)
    x = layers.Reshape((num_targets, num_classes), name="reshape_targets")(x)
    out = layers.Softmax(axis=-1, name="aoa_softmax")(x)
    return keras.Model(inp, out, name="CNN_DOA_diagram")

model = build_model(num_classes=181, num_targets=4)
model.summary()
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss,
    metrics=[keras.metrics.CategoricalAccuracy(name="cat_acc")]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=100, min_lr=1e-6),
]

history = model.fit(
    X_tr, Y_tr,
    validation_data=(X_va, Y_va),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

Yhat = model.predict(X_va, batch_size=128)      
pred_idx = np.argmax(Yhat, axis=-1)              
pred_deg = pred_idx.astype(np.float32) - 90.0 
pred_rad = pred_deg * np.pi / 180.0
true_idx = np.argmax(Y_va, axis=-1)
true_deg = true_idx.astype(np.float32) - 90.0

mae_deg = np.mean(np.abs(pred_deg - true_deg))
print("Val MAE (deg):", mae_deg)
def report_metric(pred_deg, true_deg, eps=1e-12):
    err = pred_deg - true_deg                         # (N,4)
    rmse_per_sample = np.sqrt(np.mean(err**2, axis=1)) # (N,)
    metric_per_sample = 10*np.log10(rmse_per_sample + eps)  # (N,)
    return float(np.mean(metric_per_sample)), rmse_per_sample, metric_per_sample

metric_avg, rmse_per_sample, metric_per_sample = report_metric(pred_deg, true_deg)
print("Metric = avg_n [10*log10(RMSE_n)] :", metric_avg)
