import numpy as np

def prob_to_label_classifier(pred):
	mask = (pred >= 0.5)
	x_coor, y_coor = np.where(mask)
	result = np.zeros((pred.shape[0], 6))
	for x, y in zip(x_coor, y_coor):
		loc = y // 6
		star = y % 6
		result[x][loc] = star

	return result

def prob_to_label_regressor(pred):
	result = np.zeros((pred.shape[0], 6))
	pred = pred.reshape(pred.shape[0], -1, 5)
	star = pred.argmax(axis=-1) + 1
	prob = pred.max(axis=-1)
	mask = prob >= 0.5
	result[mask] = star[mask]

	return result

def pred_to_label(outputs_classifier, outputs_regressor):

    """Convert output model to label. Get aspects have reliability >= 0.5

    Args:
        outputs_classifier (numpy.array): Output classifier layer
        outputs_regressor (numpy.array): Output regressor layer

    Returns:
        predicted label
    """

    result = np.zeros(outputs_regressor.shape)  # Initialize result with the same shape as outputs_regressor
    mask = (outputs_classifier >= 0.5)
    mask = mask.reshape(mask.shape[0], -1)  # Reshape mask to match the shape of outputs_regressor
    result[mask] = outputs_regressor[mask]
    
    return result

if __name__ == "__main__":
    np.random.seed(42)
    preds_classifier = np.random.rand(5, 30)  
    preds_regressor = np.random.rand(5, 30)

    labels_1 = prob_to_label_classifier(preds_classifier)
    print("prob_to_label_classifier Output:")
    print(labels_1)

    labels_2 = prob_to_label_regressor(preds_classifier)
    print("\nprob_to_label_regressor Output:")
    print(labels_2)

    final_labels = pred_to_label(preds_classifier, preds_regressor)
    print("\npred_to_label Output:")
    print(final_labels)