import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from LDA_Function import LDA  # Your custom LDA implementation
from LDA_MNIST import load_clean_mnist_split  # Your function to load MNIST data
from scipy.ndimage import gaussian_filter  # For Gaussian blur
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def apply_blur(X, sigma=1.0):
    """
    Applies Gaussian blur to the input MNIST images.

    Parameters:
    -----------
    X : ndarray
        Input images with shape (n_samples, 784)
    sigma : float
        Standard deviation of the Gaussian kernel, controls blur intensity

    Returns:
    --------
    ndarray
        Blurred images with same shape as input
    """
    # Reshape to 2D images
    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, 28, 28)  # MNIST is 28x28

    # Apply Gaussian blur to each image
    X_blurred = np.zeros_like(X_2d)
    for i in range(n_samples):
        X_blurred[i] = gaussian_filter(X_2d[i], sigma=sigma)

    # Reshape back to original format (flattened)
    return X_blurred.reshape(n_samples, 784)


def evaluate_with_blur(lda, knn, X_test, y_test, sigma):
    """
    Evaluate the LDA+k-NN pipeline with blurred test data.
    """
    # Apply blur to test data
    X_test_blurred = apply_blur(X_test, sigma=sigma)

    # Project blurred test data using the LDA transform
    X_test_lda_blurred = lda.transform(X_test_blurred)

    # Predict using trained k-NN model
    y_pred = knn.predict(X_test_lda_blurred)

    # Evaluate accuracy and classification report
    acc = accuracy_score(y_test, y_pred)
    misclassified = int((1 - acc) * len(y_test))  # Calculate misclassified samples

    print(f"Blur sigma: {sigma:.2f} → Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return acc, misclassified, y_pred



def visualize_blur_effects(X_sample, blur_levels):
    """
    Visualize the effect of different blur intensities on sample images.
    """
    n_samples = len(X_sample)
    n_levels = len(blur_levels)

    plt.figure(figsize=(n_levels * 3, n_samples * 3))

    for i, sample in enumerate(X_sample):
        # Original image
        plt.subplot(n_samples, n_levels, i * n_levels + 1)
        plt.imshow(sample.reshape(28, 28), cmap='gray')
        if i == 0:
            plt.title(f'Original')
        plt.axis('off')

        # Blurred images
        for j, sigma in enumerate(blur_levels[1:], 1):  # Skip the first level (0.0)
            blurred = gaussian_filter(sample.reshape(28, 28), sigma=sigma)
            plt.subplot(n_samples, n_levels, i * n_levels + j + 1)
            plt.imshow(blurred, cmap='gray')
            if i == 0:
                plt.title(f'Blur σ={sigma}')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig("blur_examples.png")
    plt.show()



def show_single_sample_across_blur_levels(sample_image, blur_levels):
    """
    Show how a single MNIST digit is affected by increasing Gaussian blur levels.
    """
    plt.figure(figsize=(len(blur_levels) * 3, 3))

    for i, sigma in enumerate(blur_levels):
        blurred = gaussian_filter(sample_image.reshape(28, 28), sigma=sigma)
        plt.subplot(1, len(blur_levels), i + 1)
        plt.imshow(blurred, cmap='gray')
        plt.title(f'σ = {sigma}')
        plt.axis('off')

    plt.suptitle('Effect of Gaussian Blur on a Single Digit Sample', fontsize=14)
    plt.tight_layout()
    plt.savefig("single_sample_blur_effect.png")
    plt.show()


def run_experiment_with_blur():
    """
    Main function to run the experiment with blur.
    """
    # Get training and testing data
    X_train, X_test, y_train, y_test = load_clean_mnist_split()

    # Ensure numpy array format
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Define regularization values and blur levels
    tau_values = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
    blur_levels = [0.0, 0.5, 1.0, 1.5, 2.0]  # Increasing levels of blur

    # Store results for CSV output
    results = []

    for tau in tau_values:
        print(f"\n===============================")
        print(f"Evaluating LDA with tau = {tau}")
        print(f"===============================\n")

        # Train LDA and kNN
        lda = LDA(n_components=9, tau=tau)
        lda.fit(X_train, y_train)
        X_train_lda = lda.transform(X_train)

        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X_train_lda, y_train)

        # Evaluate on different blur levels
        for sigma in blur_levels:
            print(f"\n--- Blur sigma = {sigma:.2f} ---")
            acc, misclassified, _ = evaluate_with_blur(lda, knn, X_test, y_test, sigma)

            # Append to results for CSV
            results.append({'tau': tau, 'sigma': sigma, 'accuracy': acc, 'misclassified': misclassified})

    # Convert results to DataFrame for easier plotting
    df_results = pd.DataFrame(results)

    # Save results to CSV
    df_results.to_csv("tau_noise_accuracy_misclassification_results.csv", index=False)

    # Plot results (same as previous code)
    plt.figure(figsize=(10, 6))
    for tau in tau_values:
        subset = df_results[df_results['tau'] == tau]
        plt.plot(subset['sigma'], subset['accuracy'], marker='o', label=f'tau={tau:.0e}')

    plt.title('Accuracy vs. Blur Intensity for Different Regularization Values')
    plt.xlabel('Blur Intensity (sigma)')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("accuracy_vs_blur.png")
    plt.show()

    # Create pivot table for heatmap: rows = tau, columns = sigma
    pivot_table = df_results.pivot_table(
        values='accuracy',
        index='tau',
        columns='sigma'
    )

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlGnBu')
    plt.title('Accuracy by Regularization (τ) and Blur Intensity (σ)')
    plt.xlabel('Blur Intensity (σ)')
    plt.ylabel('Regularization Parameter (τ)')
    plt.savefig("accuracy_heatmap_tau_vs_blur.png")
    plt.show()

    # Visualize some blurred images for reference
    visualize_blur_effects(X_test[:5], blur_levels)

    print("Visualizing blur effect on a single digit sample...")
    show_single_sample_across_blur_levels(X_test[0], blur_levels)

    return df_results


# Main execution
if __name__ == "__main__":
    results = run_experiment_with_blur()

    # Print a summary table of results
    print("\nSummary of Results:")
    total_samples = 21000  # assuming 30% of 70,000 MNIST images as test set

    for tau in results['tau'].unique():
        print(f"\nTau = {tau:.0e}")
        subset = results[results['tau'] == tau]
        for _, row in subset.iterrows():
            misclassified = row['misclassified']
            print(f"  Blur σ={row['sigma']:.2f} → Accuracy={row['accuracy'] * 100:.2f}%, Misclassified: {misclassified}")

