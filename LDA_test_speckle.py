import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from LDA_Function import LDA
from LDA_MNIST import load_clean_mnist_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def add_speckle_noise(X, std=0.1):
    noise = np.random.normal(0, std, X.shape)
    return np.clip(X + X * noise, 0.0, 1.0)


def evaluate_with_speckle_noise(lda, knn, X_test, y_test, noise_std):
    """
    Evaluate the LDA+k-NN pipeline with speckle noise added to test data.
    """
    # Apply speckle noise to test data
    X_test_noisy = add_speckle_noise(X_test, std=noise_std)

    # Project noisy test data using the LDA transform
    X_test_lda_noisy = lda.transform(X_test_noisy)

    # Predict using trained k-NN model
    y_pred = knn.predict(X_test_lda_noisy)

    # Evaluate accuracy and classification report
    acc = accuracy_score(y_test, y_pred)
    print(f"Speckle noise STD: {noise_std:.2f} → Accuracy: {acc * 100:.2f}%")

    # Print classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return acc, y_pred, cm


def visualize_speckle_noise_effects(X_sample, noise_levels):
    """
    Visualize the effect of different speckle noise levels on sample images.
    """
    n_samples = len(X_sample)
    n_levels = len(noise_levels)

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
        for j, std in enumerate(noise_levels[1:], 1):  # Skip the first level (0.0)
            # Add noise
            noisy = add_speckle_noise(sample.reshape(1, -1), std=std)

            plt.subplot(n_samples, n_levels, i * n_levels + j + 1)
            plt.imshow(noisy.reshape(28, 28), cmap='gray')
            if i == 0:
                plt.title(f'Noise σ={std}')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig("speckle_noise_examples.png")
    plt.show()

def show_single_sample_speckle_noise(sample_image, noise_levels):
    """
    Visualize how a single image looks under different levels of speckle noise.
    """
    plt.figure(figsize=(len(noise_levels) * 3, 3))

    for i, std in enumerate(noise_levels):
        noisy = add_speckle_noise(sample_image.reshape(1, -1), std=std)
        plt.subplot(1, len(noise_levels), i + 1)
        plt.imshow(noisy.reshape(28, 28), cmap='gray')
        plt.title(f"σ={std}")
        plt.axis('off')

    plt.suptitle("Effect of Speckle Noise on a Single Digit Sample", fontsize=14)
    plt.tight_layout()
    plt.savefig("single_sample_speckle_noise.png")
    plt.show()


def run_speckle_noise_experiment():
    """
    Main function to run the experiment with speckle noise.
    """
    # Get training and testing data
    X_train, X_test, y_train, y_test = load_clean_mnist_split()

    # Ensure numpy array format
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Define regularization values and noise levels
    tau_values = [1e-12 ,1e-10, 1e-8, 1e-6, 1e-4]
    noise_levels = [0.0, 0.1, 0.2, 0.4, 0.8]  # Increasing levels of noise

    # Store results for plotting
    results = []
    confusion_matrices = {}

    # Visualize some noisy images for reference
    print("Visualizing examples of speckle noise effects...")
    visualize_speckle_noise_effects(X_test[:5], noise_levels)

    print("Visualizing speckle effect on a single digit sample...")
    show_single_sample_speckle_noise(X_test[0], noise_levels)

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
        for noise_std in noise_levels:
            print(f"\n--- Speckle Noise STD = {noise_std:.2f} ---")
            acc, y_pred, cm = evaluate_with_speckle_noise(lda, knn, X_test, y_test, noise_std)
            n_misclassified = np.sum(y_test != y_pred)
            results.append({
                'tau': tau,
                'noise_std': noise_std,
                'accuracy': acc,
                'misclassified': n_misclassified
            })

            confusion_matrices[(tau, noise_std)] = cm

    # Convert results to DataFrame for easier plotting
    df_results = pd.DataFrame(results)

    # Plot results
    plt.figure(figsize=(10, 6))
    for tau in tau_values:
        subset = df_results[df_results['tau'] == tau]
        plt.plot(subset['noise_std'], subset['accuracy'], marker='o', label=f'tau={tau:.0e}')

    plt.title('Accuracy vs. Speckle Noise Level for Different Regularization Values')
    plt.xlabel('Noise Level (standard deviation)')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("accuracy_vs_speckle_noise.png")
    plt.show()

    # Plot heatmap of results
    pivot_table = df_results.pivot_table(
        values='accuracy',
        index='tau',
        columns='noise_std'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlGnBu')
    plt.title('Accuracy by Regularization (tau) and Speckle Noise Level')
    plt.xlabel('Noise Level (standard deviation)')
    plt.ylabel('Regularization Parameter (tau)')
    plt.savefig("accuracy_heatmap_tau_vs_speckle_noise.png")
    plt.show()

    # Save results to CSV
    df_results.to_csv("speckle_noise_results.csv", index=False)

    return df_results


# Main execution
if __name__ == "__main__":
    # Set fixed random seed for reproducibility
    np.random.seed(42)

    # Run the main speckle noise experiment
    results = run_speckle_noise_experiment()

    # Print a summary table
    print("\nSummary of Results:")
    for tau in results['tau'].unique():
        print(f"\nTau = {tau:.0e}")
        subset = results[results['tau'] == tau]
        for _, row in subset.iterrows():
            print(f"  Noise σ={row['noise_std']:.2f} → Accuracy={row['accuracy'] * 100:.2f}%")