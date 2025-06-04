import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from LDA_Function import LDA  # Your custom LDA implementation
from LDA_MNIST import load_clean_mnist_split  # Your function to load MNIST data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def apply_salt_pepper_noise(X, density=0.05):
    """
    Applies salt and pepper noise to the input MNIST images.

    Parameters:
    -----------
    X : ndarray
        Input images with shape (n_samples, 784)
    density : float
        Proportion of the image pixels to be replaced with noise (salt or pepper)
        Salt noise = white pixels (1.0)
        Pepper noise = black pixels (0.0)

    Returns:
    --------
    ndarray
        Noisy images with same shape as input
    """
    # Create a copy of the input images
    X_noisy = X.copy()

    # Generate random indices for the pixels to be affected by noise
    n_samples, n_pixels = X.shape
    n_noise_pixels = int(density * n_pixels)

    for i in range(n_samples):
        # Random indices for noise pixels
        noise_idx = np.random.choice(n_pixels, n_noise_pixels, replace=False)

        # Randomly choose which indices will be salt (1) and which will be pepper (0)
        salt_idx = np.random.choice(noise_idx, size=n_noise_pixels // 2, replace=False)
        pepper_idx = np.setdiff1d(noise_idx, salt_idx)

        # Apply salt noise (white pixels)
        X_noisy[i, salt_idx] = 1.0

        # Apply pepper noise (black pixels)
        X_noisy[i, pepper_idx] = 0.0

    return X_noisy


def evaluate_with_salt_pepper_noise(lda, knn, X_test, y_test, density):
    """
    Evaluate the LDA+k-NN pipeline with salt and pepper noise added to test data.
    """
    # Apply salt and pepper noise to test data
    X_test_noisy = apply_salt_pepper_noise(X_test, density=density)

    # Project noisy test data using the LDA transform
    X_test_lda_noisy = lda.transform(X_test_noisy)

    # Predict using trained k-NN model
    y_pred = knn.predict(X_test_lda_noisy)

    # Evaluate accuracy and classification report
    acc = accuracy_score(y_test, y_pred)
    print(f"Salt & Pepper noise density: {density:.3f} → Accuracy: {acc * 100:.2f}%")

    # Print classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return acc, y_pred, cm, (len(y_test) - np.sum(y_test == y_pred))


def visualize_salt_pepper_effects(X_sample, noise_densities):
    """
    Visualize the effect of different salt and pepper noise levels on sample images.
    """
    n_samples = len(X_sample)
    n_levels = len(noise_densities)

    # Set a fixed random seed for reproducible visualization
    np.random.seed(42)

    plt.figure(figsize=(n_levels * 3, n_samples * 3))

    for i, sample in enumerate(X_sample):
        # Original image
        plt.subplot(n_samples, n_levels, i * n_levels + 1)
        plt.imshow(sample.reshape(28, 28), cmap='gray')
        if i == 0:
            plt.title(f'Original')
        plt.axis('off')

        # Noisy images
        for j, density in enumerate(noise_densities[1:], 1):  # Skip the first level (0.0)
            # Create a copy for this example
            noisy = sample.copy()

            # Calculate number of pixels to affect
            n_pixels = sample.size
            n_noise_pixels = int(density * n_pixels)

            # Random indices for noise pixels
            noise_idx = np.random.choice(n_pixels, n_noise_pixels, replace=False)

            # Randomly choose which indices will be salt (1) and which will be pepper (0)
            salt_idx = np.random.choice(noise_idx, size=n_noise_pixels // 2, replace=False)
            pepper_idx = np.setdiff1d(noise_idx, salt_idx)

            # Apply salt noise (white pixels)
            noisy.flat[salt_idx] = 1.0

            # Apply pepper noise (black pixels)
            noisy.flat[pepper_idx] = 0.0

            plt.subplot(n_samples, n_levels, i * n_levels + j + 1)
            plt.imshow(noisy.reshape(28, 28), cmap='gray')
            if i == 0:
                plt.title(f'Noise d={density:.2f}')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig("salt_pepper_noise_examples.png")
    plt.show()
def show_single_sample_across_densities(sample_image, noise_densities):
    """
    Show how a single MNIST digit is affected by increasing salt-and-pepper noise levels.
    """
    plt.figure(figsize=(len(noise_densities) * 3, 3))

    for i, d in enumerate(noise_densities):
        noisy_sample = apply_salt_pepper_noise(sample_image.reshape(1, -1), density=d)
        plt.subplot(1, len(noise_densities), i + 1)
        plt.imshow(noisy_sample.reshape(28, 28), cmap='gray')
        plt.title(f'd = {d:.2f}')
        plt.axis('off')

    plt.suptitle('Impact of Salt-and-Pepper Noise on a Single Digit Image', fontsize=14)
    plt.tight_layout()
    plt.savefig("salt_pepper_single_sample.png")
    plt.show()


def run_salt_pepper_experiment():
    """
    Main function to run the experiment with salt and pepper noise.
    """
    # Get training and testing data
    X_train, X_test, y_train, y_test = load_clean_mnist_split()

    # Ensure numpy array format
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Define regularization values and noise densities
    tau_values = [1e-12 ,1e-10, 1e-8, 1e-6, 1e-4]
    noise_densities = [0.0, 0.05, 0.1, 0.2, 0.3]  # Proportion of pixels affected by noise

    # Store results for plotting
    results = []
    confusion_matrices = {}

    # Visualize some noisy images for reference
    print("Visualizing examples of salt and pepper noise effects...")
    visualize_salt_pepper_effects(X_test[:5], noise_densities)

    # Show noise effect on a single digit image
    print("Showing salt-and-pepper noise effect on a single digit sample...")
    show_single_sample_across_densities(X_test[0], noise_densities)

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

        # Evaluate on different noise levels
        for density in noise_densities:
            print(f"\n--- Salt & Pepper Noise Density = {density:.2f} ---")
            acc, y_pred, cm, misclassified = evaluate_with_salt_pepper_noise(lda, knn, X_test, y_test, density)
            results.append({
                'tau': tau,
                'density': density,
                'accuracy': acc,
                'misclassified': misclassified
            })

            confusion_matrices[(tau, density)] = cm

    # Convert results to DataFrame for easier plotting
    df_results = pd.DataFrame(results)

    # Plot results
    plt.figure(figsize=(10, 6))
    for tau in tau_values:
        subset = df_results[df_results['tau'] == tau]
        plt.plot(subset['density'], subset['accuracy'], marker='o', label=f'tau={tau:.0e}')

    plt.title('Accuracy vs. Salt & Pepper Noise Density for Different Regularization Values')
    plt.xlabel('Noise Density (proportion of affected pixels)')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("accuracy_vs_salt_pepper_noise.png")
    plt.show()

    # Plot heatmap of results
    pivot_table = df_results.pivot_table(
        values='accuracy',
        index='tau',
        columns='density'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlGnBu')
    plt.title('Accuracy by Regularization (tau) and Salt & Pepper Noise Density')
    plt.xlabel('Noise Density')
    plt.ylabel('Regularization Parameter (tau)')
    plt.savefig("accuracy_heatmap_tau_vs_salt_pepper.png")
    plt.show()

    # Compare different noise types if available (optional)
    try:
        # Try to load results from gaussian noise experiment, if available
        gaussian_results = pd.read_csv("gaussian_noise_results.csv")

        # Compute mean accuracy for each tau value at medium noise level
        medium_density = 0.1  # or choose another comparable value
        medium_std = 0.1  # equivalent gaussian noise level

        salt_pepper_medium = df_results[df_results['density'] == medium_density].groupby('tau')['accuracy'].mean()
        gaussian_medium = gaussian_results[gaussian_results['noise_std'] == medium_std].groupby('tau')[
            'accuracy'].mean()

        # Create comparison plot
        plt.figure(figsize=(8, 6))
        tau_list = sorted(tau_values)
        plt.plot(tau_list, [salt_pepper_medium[tau] for tau in tau_list], 'o-', label='Salt & Pepper')
        plt.plot(tau_list, [gaussian_medium[tau] for tau in tau_list], 's-', label='Gaussian')
        plt.xscale('log')
        plt.title(f'Salt & Pepper vs. Gaussian Noise at Medium Noise Level')
        plt.xlabel('Regularization Parameter (tau)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("noise_type_comparison.png")
        plt.show()
    except:
        print("Could not generate noise type comparison (gaussian_noise_results.csv not found)")

    # Save results to CSV
    df_results.to_csv("salt_pepper_noise_results.csv", index=False)

    return df_results


# Main execution
if __name__ == "__main__":
    results = run_salt_pepper_experiment()

    # Print a summary table
    print("\nSummary of Results:")
    for tau in results['tau'].unique():
        print(f"\nTau = {tau:.0e}")
        subset = results[results['tau'] == tau]
        for _, row in subset.iterrows():
            print(f"  Noise density={row['density']:.2f} → Accuracy={row['accuracy'] * 100:.2f}%, Misclassified: {int(row['misclassified'])}")
