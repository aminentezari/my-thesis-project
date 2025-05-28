import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from LDA_Function import LDA
from LDA_MNIST import load_clean_mnist_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import ndimage


def add_zigzag_noise(X, amplitude=2, frequency=2):
    """
    Apply zigzag distortion to images, simulating scanning artifacts or interference.

    Parameters:
    -----------
    X : ndarray
        Input images with shape (n_samples, 784)
    amplitude : float
        Intensity of the zigzag displacement (higher = stronger distortion)
    frequency : float
        Number of zigzag cycles across the image (higher = more zigzags)

    Returns:
    --------
    ndarray
        Distorted images with same shape as input
    """
    # Create output array
    X_zigzag = np.zeros_like(X)

    # Reshape to 2D images
    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, 28, 28)

    # Process each image
    for i in range(n_samples):
        image = X_2d[i]
        distorted = np.zeros_like(image)

        # Create horizontal zigzag (row shifts)
        for y in range(28):
            # Calculate horizontal shift based on sine wave
            shift = amplitude * np.sin(2 * np.pi * frequency * y / 28)

            # Apply the shift using interpolation
            for x in range(28):
                x_shifted = x + shift

                # Ensure the shifted coordinate is within bounds
                if 0 <= x_shifted < 27:
                    # Linear interpolation between two nearest pixels
                    x_floor = int(np.floor(x_shifted))
                    x_ceil = int(np.ceil(x_shifted))
                    dx = x_shifted - x_floor

                    # Weighted average of two pixels
                    distorted[y, x] = (1 - dx) * image[y, x_floor] + dx * image[y, x_ceil]
                elif x_shifted >= 27:
                    distorted[y, x] = image[y, 27]
                else:
                    distorted[y, x] = image[y, 0]

        # Store the distorted image
        X_zigzag[i] = distorted.flatten()

    return X_zigzag


def evaluate_with_zigzag_noise(lda, knn, X_test, y_test, amplitude, frequency):
    """
    Evaluate the LDA+k-NN pipeline with zigzag distortion added to test data.
    """
    # Apply zigzag noise to test data
    X_test_noisy = add_zigzag_noise(X_test, amplitude=amplitude, frequency=frequency)

    # Project noisy test data using the LDA transform
    X_test_lda_noisy = lda.transform(X_test_noisy)

    # Predict using trained k-NN model
    y_pred = knn.predict(X_test_lda_noisy)

    # Evaluate accuracy and classification report
    acc = accuracy_score(y_test, y_pred)
    print(f"Zigzag noise (amplitude={amplitude:.1f}, frequency={frequency:.1f}) → Accuracy: {acc * 100:.2f}%")

    # Print classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    n_misclassified = len(y_test) - np.sum(y_test == y_pred)
    return acc, y_pred, cm, n_misclassified
def show_amplitude_across_samples(X_samples, amplitudes, fixed_freq=2):
    n_samples = len(X_samples)
    n_params = len(amplitudes)

    plt.figure(figsize=(n_params * 2.5, n_samples * 2.5))
    for i, sample in enumerate(X_samples):
        for j, amp in enumerate(amplitudes):
            distorted = add_zigzag_noise(sample.reshape(1, -1), amplitude=amp, frequency=fixed_freq)
            plt.subplot(n_samples, n_params, i * n_params + j + 1)
            plt.imshow(distorted.reshape(28, 28), cmap='gray')
            if i == 0:
                plt.title(f"A={amp}")
            if j == 0:
                plt.ylabel(f"Sample {i+1}")
            plt.axis('off')

    plt.suptitle(f"Effect of Amplitude (F={fixed_freq}) on Multiple Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig("amplitude_effect_multiple_samples.png")
    plt.show()



def visualize_confusion_matrices(confusion_matrices):
    """
    Visualizes the confusion matrices for different tau and zigzag parameter combinations.
    """
    for (tau, param_key), cm in confusion_matrices.items():
        plt.figure(figsize=(8, 6))

        # Plot heatmap for confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))

        plt.title(f'Confusion Matrix for Tau={tau} and Zigzag Params={param_key}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(f"confusion_matrix_tau_{tau}_params_{param_key}.png")
        plt.show()

def visualize_zigzag_noise_effects(X_sample, params):
    """
    Visualize the effect of different zigzag distortion parameters on sample images.
    """
    n_samples = len(X_sample)
    n_params = len(params)

    # Set a fixed random seed for reproducible visualization
    np.random.seed(42)

    plt.figure(figsize=(n_params * 3, n_samples * 3))

    for i, sample in enumerate(X_sample):
        # Original image
        plt.subplot(n_samples, n_params, i * n_params + 1)
        plt.imshow(sample.reshape(28, 28), cmap='gray')
        if i == 0:
            plt.title(f'Original')
        plt.axis('off')

        # Distorted images
        for j, (amplitude, frequency) in enumerate(params[1:], 1):  # Skip the first param pair
            # Apply zigzag to a single image
            sample_reshaped = sample.reshape(1, -1)
            distorted = add_zigzag_noise(sample_reshaped, amplitude=amplitude, frequency=frequency)

            plt.subplot(n_samples, n_params, i * n_params + j + 1)
            plt.imshow(distorted.reshape(28, 28), cmap='gray')
            if i == 0:
                plt.title(f'A={amplitude}, F={frequency}')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig("zigzag_noise_examples.png")
    plt.show()

def show_single_sample_across_zigzag_params(sample_image, zigzag_params):
    """
    Show how a single MNIST digit is distorted by different zigzag amplitude and frequency values.
    """
    plt.figure(figsize=(len(zigzag_params) * 3, 3))

    for i, (amp, freq) in enumerate(zigzag_params):
        distorted = add_zigzag_noise(sample_image.reshape(1, -1), amplitude=amp, frequency=freq)
        plt.subplot(1, len(zigzag_params), i + 1)
        plt.imshow(distorted.reshape(28, 28), cmap='gray')
        plt.title(f"A={amp}, F={freq}")
        plt.axis('off')

    plt.suptitle("Effect of Zigzag Noise on a Single Digit Sample", fontsize=14)
    plt.tight_layout()
    plt.savefig("single_sample_zigzag_noise.png")
    plt.show()


def run_zigzag_noise_experiment():
    """
    Main function to run the experiment with zigzag noise.
    """
    # Get training and testing data
    X_train, X_test, y_train, y_test = load_clean_mnist_split()

    # Ensure numpy array format
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Define regularization values and zigzag parameters (amplitude, frequency)
    tau_values = [1e-12 ,1e-10, 1e-8, 1e-6, 1e-4]
    zigzag_params = [
        (0, 0),  # No distortion (original images)
        (1, 2),  # Mild distortion
        (2, 2),  # Medium amplitude, medium frequency
        (3, 2),  # Higher amplitude, medium frequency
        (2, 4)  # Medium amplitude, higher frequency
    ]

    # Store results for plotting
    results = []
    confusion_matrices = {}

    # Visualize zigzag distortion examples
    print("Visualizing examples of zigzag distortion...")
    visualize_zigzag_noise_effects(X_test[:5], zigzag_params)

    # Show single sample with different zigzag settings
    print("Visualizing zigzag effect on a single digit sample...")
    show_single_sample_across_zigzag_params(X_test[0], zigzag_params)

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

        # Evaluate on different zigzag parameters
        for amplitude, frequency in zigzag_params:
            print(f"\n--- Zigzag parameters: amplitude={amplitude:.1f}, frequency={frequency:.1f} ---")

            # Correct function call with all arguments
            acc, y_pred, cm, misclassified = evaluate_with_zigzag_noise(
                lda, knn, X_test, y_test, amplitude, frequency
            )

            param_key = f"{amplitude}_{frequency}"

            results.append({
                'tau': tau,
                'param_key': param_key,
                'amplitude': amplitude,
                'frequency': frequency,
                'accuracy': acc,
                'misclassified': misclassified
            })

            confusion_matrices[(tau, param_key)] = cm



    # Convert results to DataFrame for easier plotting
    df_results = pd.DataFrame(results)

    # Plot results - we'll use amplitude as main factor for plotting
    plt.figure(figsize=(10, 6))

    # Group by frequency for better visualization
    freq_groups = df_results['frequency'].unique()

    for tau in tau_values:
        for freq in freq_groups:
            subset = df_results[(df_results['tau'] == tau) & (df_results['frequency'] == freq)]
            if len(subset) > 1:  # Only plot if we have points to connect
                plt.plot(subset['amplitude'], subset['accuracy'],
                         marker='o',
                         label=f'tau={tau:.0e}, freq={freq}')

    plt.title('Accuracy vs. Zigzag Amplitude for Different Regularization Values')
    plt.xlabel('Zigzag Amplitude')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("accuracy_vs_zigzag.png")
    plt.show()

    # Create heatmap focusing on amplitude
    # Filter for frequency=2 (the most common in our params)
    common_freq = 2
    pivot_amplitude = df_results[df_results['frequency'] == common_freq].pivot_table(
        values='accuracy',
        index='tau',
        columns='amplitude'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_amplitude, annot=True, fmt='.3f', cmap='YlGnBu')
    plt.title(f'Accuracy by Regularization (tau) and Zigzag Amplitude (frequency={common_freq})')
    plt.xlabel('Zigzag Amplitude')
    plt.ylabel('Regularization Parameter (tau)')
    plt.savefig("accuracy_heatmap_tau_vs_zigzag.png")
    plt.show()

    fixed_amplitude = 2
    pivot_frequency = df_results[df_results['amplitude'] == fixed_amplitude].pivot_table(
        values='accuracy',
        index='tau',
        columns='frequency'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_frequency, annot=True, fmt='.3f', cmap='Greys')
    plt.title(f'Accuracy by Regularization (tau) and Zigzag Frequency (amplitude={fixed_amplitude})')
    plt.xlabel('Zigzag Frequency')
    plt.ylabel('Regularization Parameter (tau)')
    plt.savefig("accuracy_heatmap_tau_vs_frequency.png")
    plt.show()


    # Line plot: Accuracy vs. Zigzag Frequency for different tau values (fixed amplitude)
    fixed_amplitude = 2
    plt.figure(figsize=(10, 6))

    freq_values = sorted(df_results['frequency'].unique())
    for tau in tau_values:
        subset = df_results[(df_results['tau'] == tau) & (df_results['amplitude'] == fixed_amplitude)]
        if len(subset) > 1:
            plt.plot(subset['frequency'], subset['accuracy'],
                     marker='o',
                     label=f'tau={tau:.0e}')

    plt.title(f'Accuracy vs. Zigzag Frequency (Amplitude={fixed_amplitude})')
    plt.xlabel('Zigzag Frequency')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("accuracy_vs_frequency.png")
    plt.show()



    # Save results to CSV
    df_results.to_csv("zigzag_noise_results.csv", index=False)

    return df_results


# Main execution
if __name__ == "__main__":
    # Run the main zigzag noise experiment
    results = run_zigzag_noise_experiment()

    # Print a summary table
    print("\nSummary of Results:")
    for tau in results['tau'].unique():
        print(f"\nTau = {tau:.0e}")
        subset = results[results['tau'] == tau]
        for _, row in subset.iterrows():
            print(
                f"  Zigzag: amplitude={row['amplitude']}, frequency={row['frequency']} → "
                f"Accuracy={row['accuracy'] * 100:.2f}%, Misclassified: {int(row['misclassified'])}")
