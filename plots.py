import matplotlib.pyplot as plt
palette = [
'midnightblue',
'darkslateblue',
'slateblue',
'indigo',
'mediumpurple',
'orchid'
]
# -----------------------------
# Architecture Experiments
# -----------------------------


# Depth Experiment
# -----------------------------

depth_arch = ["[128]", "[256,128]", "[512,256,128]"]

depth_acc = [
97.87,
98.43,
98.55
]

plt.figure(figsize=(8,4))

plt.bar(depth_arch, depth_acc, color=palette[:3])

plt.ylabel("Test Accuracy (%)")
plt.xlabel("MLP Depth (Hidden Layer Configuration)")
plt.title("Impact of Network Depth on Model Accuracy")

plt.ylim(97.8, 98.6)

plt.grid(axis="y")

plt.tight_layout()
plt.savefig("depth_experiment.png")

plt.show()

# Width Experiment
# -----------------------------

width_arch = ["[128,128]", "[512,512]"]

width_acc = [
98.16,
98.49
]

plt.figure(figsize=(6,4))

plt.bar(width_arch, width_acc, color=[palette[4], palette[5]])

plt.ylabel("Test Accuracy (%)")
plt.xlabel("Hidden Layer Width")
plt.title("Impact of Hidden Layer Width on Model Accuracy")

plt.ylim(98.1, 98.6)

plt.grid(axis="y")

plt.tight_layout()
plt.savefig("width_experiment.png")

plt.show()


# -----------------------------
# Dropout Experiments
# -----------------------------

dropout_rates = [0, 0.3, 0.5]

acc_dropout = [
    98.44,
    98.55,
    98.36
]

plt.figure()
plt.plot(dropout_rates, acc_dropout, marker='*', linewidth=2, color=palette[0])
plt.xlabel("Dropout Rate")
plt.ylabel("Test Accuracy (%)")
plt.title("Impact of Dropout Rate on Model Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("dropout_vs_accuracy.png")
plt.show()


# -----------------------------
# Regularization Experiments
# -----------------------------

regularizers = ["None", "L2 (1e-4)", "L2 (1e-3)", "L1 (1e-5)"]
acc_reg_type = [98.61, 98.55, 98.23, 98.47]

plt.figure()
plt.bar(regularizers, acc_reg_type, color=palette[:len(regularizers)])

plt.ylim(97.8, 98.7)

plt.grid(axis="y")

plt.ylabel("Test Accuracy (%)")
plt.title("Comparison of Regularization Methods")
plt.savefig("regularization_types.png")
plt.show()

# -----------------------------
# Activation Function Comparison
# -----------------------------

activations = ["ReLU", "GELU"]

acc_act = [
    98.55,
    98.43
]

plt.figure()
plt.bar(activations, acc_act, color=[palette[0], palette[3]])

plt.ylim(97.8, 98.7)

plt.grid(axis="y")

plt.ylabel("Test Accuracy (%)")
plt.title("Activation Function Comparison")
plt.tight_layout()
plt.savefig("activation_comparison.png")
plt.show()


# -----------------------------
# Training vs Validation Loss
# -----------------------------

epochs = list(range(1, 11))

train_loss = [
0.2930,
0.1502,
0.1230,
0.1099,
0.0953,
0.0703,
0.0608,
0.0605,
0.0540,
0.0525
]

val_loss = [
0.1073,
0.0854,
0.0751,
0.0680,
0.0558,
0.0497,
0.0515,
0.0498,
0.0491,
0.0520
]

plt.figure()

plt.plot(epochs, train_loss, marker='*', linewidth=2, label="Training Loss", color=palette[3])
plt.plot(epochs, val_loss, marker='*', linewidth=2, label="Validation Loss", color=palette[4])

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss over Epochs")

plt.legend()
plt.grid(True)

plt.savefig("training_validation_loss.png")
plt.show()


# -----------------------------
# With and Without BatchNorm
# -----------------------------

bn_labels = ["With BatchNorm", "Without BatchNorm"]
bn_acc = [98.55, 98.40]

plt.figure()
plt.bar(bn_labels, bn_acc, color=[palette[2], palette[5]])

plt.ylim(97.8, 98.7)

plt.grid(axis="y")

plt.ylabel("Test Accuracy (%)")
plt.title("Impact of Batch Normalization")
plt.savefig("batchnorm_comparison.png")
plt.show()