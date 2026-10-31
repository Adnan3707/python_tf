"""Reusable plotting helpers for model training."""

import matplotlib.pyplot as plt


def plot_loss_curves(history):
	"""Plot loss and accuracy curves from a Keras History object."""
	loss = history.history["loss"]
	val_loss = history.history["val_loss"]
	accuracy = history.history.get("accuracy", history.history.get("acc"))
	val_accuracy = history.history.get(
		"val_accuracy", history.history.get("val_acc")
	)

	if accuracy is None or val_accuracy is None:
		raise ValueError("History must contain training and validation accuracy.")

	epochs = range(len(loss))
	plt.figure(figsize=(8, 8))

	plt.subplot(2, 1, 1)
	plt.plot(epochs, loss, label="training_loss")
	plt.plot(epochs, val_loss, label="validation_loss")
	plt.title("Loss")
	plt.xlabel("Epochs")
	plt.legend()

	plt.subplot(2, 1, 2)
	plt.plot(epochs, accuracy, label="training_accuracy")
	plt.plot(epochs, val_accuracy, label="validation_accuracy")
	plt.title("Accuracy")
	plt.xlabel("Epochs")
	plt.legend()

	plt.tight_layout()
	plt.show()


__all__ = ["plot_loss_curves"]
