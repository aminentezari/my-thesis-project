import numpy as np
from scipy import linalg
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.utils import resample
from LDA_Function import LDA


# Function to add Gaussian noise to images
def add_gaussian_noise(data, mean=0, std=0.1):
    """
    Add Gaussian noise to data.

    Parameters:
    -----------
    data : array-like
        Input data.
    mean : float, default=0
        Mean of the Gaussian noise.
    std : float, default=0.1
        Standard deviation of the Gaussian noise.

    Returns:
    --------
    noisy_data : array-like
        Data with added Gaussian noise.
    """
    np.random.seed(42)  # For reproducible noise
    noise = np.random.normal(mean, std, data.shape)
    noisy_data = data + noise
    return np.clip(noisy_data, 0, 16)  # Clip to valid range for MNIST (0-16)


# Function to visualize more samples at different noise levels
def visualize_noise_samples(X, y, noise_levels, n_samples=10, n_rows=2):
    """
    Visualize multiple samples at different noise levels

    Parameters:
    -----------
    X : array-like
        Original image data
    y : array-like
        Labels for the images
    noise_levels : list
        List of noise levels to visualize
    n_samples : int, default=10
        Number of samples to show per noise level
    n_rows : int, default=2
        Number of rows of samples to show
    """
    n_cols = n_samples // n_rows

    # Create a figure for all noise levels
    plt.figure(figsize=(20, 5 * len(noise_levels)))

    # For each noise level
    for i, noise_level in enumerate(noise_levels):
        # Add the noise to the data
        X_noisy = add_gaussian_noise(X.copy(), std=noise_level)

        plt.subplot(len(noise_levels), 1, i + 1)
        plt.suptitle(f'Sample Images at Different Noise Levels', fontsize=20, y=0.98)

        # Create a subplot grid for this noise level
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        fig.suptitle(f'Digit Samples with Noise Level = {noise_level}', fontsize=16)

        # Flatten the axes array for easier indexing
        axes = axes.flatten()

        # For each sample
        for j in range(n_samples):
            # Get random index, but ensure we have a good distribution of digits
            digit_idx = (j % 10) + (j // 10) * 100  # Sample different digits

            # Show the noisy image
            axes[j].imshow(X_noisy[digit_idx].reshape(8, 8), cmap='gray')
            axes[j].set_title(f'Digit: {y[digit_idx]}', fontsize=12)
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'noise_level_samples_{noise_level}.png')
        plt.close()

    # Create a comparison plot with one example from each class for all noise levels
    classes = np.unique(y)
    fig, axes = plt.subplots(len(classes), len(noise_levels), figsize=(3 * len(noise_levels), 2.5 * len(classes)))
    fig.suptitle(f'Comparison of All Digits Across Noise Levels', fontsize=16)

    # Set column titles (noise levels)
    for j, noise_level in enumerate(noise_levels):
        axes[0, j].set_title(f'Noise = {noise_level}', fontsize=12)

    # For each class (digit)
    for i, digit in enumerate(classes):
        # Find first occurrence of this digit
        idx = np.where(y == digit)[0][0]

        # Set row labels (digit classes)
        axes[i, 0].set_ylabel(f'Digit {digit}', fontsize=12, rotation=0, labelpad=40,
                              va='center', ha='right')

        # For each noise level
        for j, noise_level in enumerate(noise_levels):
            # Create noisy version
            X_noisy = add_gaussian_noise(X.copy(), std=noise_level)

            # Show the image
            axes[i, j].imshow(X_noisy[idx].reshape(8, 8), cmap='gray')
            axes[i, j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('digit_noise_comparison.png')
    plt.close()


# Function to visualize decision boundaries
def visualize_decision_boundaries_fixed(X, y, noise_levels, tau_values, seed=42):
    """
    Simplified version of decision boundary visualization focused on digits 0-2.

    Parameters:
    -----------
    X : array-like
        Original image data
    y : array-like
        Labels for the images
    noise_levels : list
        List of noise levels to visualize
    tau_values : list
        List of regularization parameters to visualize
    seed : int, default=42
        Random seed for reproducibility
    """
    np.random.seed(seed)

    # Focus on just digits 0, 1, and 2 for clearer visualization
    selected_classes = [0, 1, 2]
    mask = np.isin(y, selected_classes)
    X_subset = X[mask]
    y_subset = y[mask]

    # Only use 2 components (maximum for 3 classes)
    n_components = 2

    # Create combined visualization in a grid
    fig, axes = plt.subplots(len(noise_levels), len(tau_values), figsize=(15, 15))
    fig.suptitle('Effect of Noise and Regularization on Decision Boundaries', fontsize=16)

    # Create noisy versions of the data
    X_noisy_dict = {}
    for noise_level in noise_levels:
        X_noisy_dict[noise_level] = add_gaussian_noise(X_subset.copy(), std=noise_level)

    # Use smaller mesh step for better resolution
    h = 0.2

    # Plot all combinations
    for i, noise_level in enumerate(noise_levels):
        for j, tau in enumerate(tau_values):
            print(f"Processing noise={noise_level}, tau={tau}")
            ax = axes[i, j]

            try:
                # Create and fit LDA with exactly 2 components
                lda = LDA(n_components=n_components, tau=tau)
                X_noisy = X_noisy_dict[noise_level]
                lda.fit(X_noisy, y_subset)

                # Transform the data to LDA space
                X_lda = lda.transform(X_noisy)

                # Train a classifier on the transformed data
                classifier = KNeighborsClassifier(n_neighbors=5)
                classifier.fit(X_lda, y_subset)

                # Create a mesh grid for visualization
                x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
                y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

                # Predict class for each mesh point - reshape for classifier input
                Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                # Plot the decision boundary
                ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

                # Plot the data points with different markers and colors
                colors = ['red', 'blue', 'green']
                markers = ['o', 's', '^']

                for idx, label in enumerate(selected_classes):
                    class_mask = y_subset == label
                    ax.scatter(X_lda[class_mask, 0], X_lda[class_mask, 1],
                               c=colors[idx], marker=markers[idx], s=30, alpha=0.8,
                               edgecolor='k', label=f'Digit {label}')

                ax.set_title(f'Noise={noise_level}, τ={tau}')

                # Only add legend to the first subplot
                if i == 0 and j == 0:
                    ax.legend()

            except Exception as e:
                print(f"Error: {str(e)}")
                ax.text(0.5, 0.5, f"Error\n{str(e)}",
                        ha='center', va='center', transform=ax.transAxes)

            # Add axis labels
            if i == len(noise_levels) - 1:  # Bottom row
                ax.set_xlabel('Component 1')
            if j == 0:  # Leftmost column
                ax.set_ylabel('Component 2')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig('decision_boundaries_fixed.png', dpi=150)
    plt.close(fig)

    print("Decision boundary visualization complete.")


# Main experiment function to test the relationship between noise and regularization
def run_noise_reg_experiment(n_samples=1000, n_components=4, seed=42):
    """
    Run an experiment to evaluate the relationship between noise levels and
    regularization parameters in LDA classification of MNIST digits.

    Parameters:
    -----------
    n_samples : int, default=1000
        Total number of samples to use (balanced across classes).
    n_components : int, default=4
        Number of LDA components to use.
    seed : int, default=42
        Random seed for reproducibility.

    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing all experiment results.
    """
    # Define noise levels to test
    noise_levels = [0, 0.1, 0.2, 0.5, 1.0, 2.0]

    # Define regularization parameters to test (tau values)
    tau_values = [0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]

    # Create a DataFrame to store all results
    results_data = []

    # Load and prepare data once to avoid reloading for each experiment
    # Set a fixed random seed for reproducibility
    np.random.seed(seed)

    # Load the MNIST dataset
    print("Loading MNIST dataset...")
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    # Split the original data to ensure consistent test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    # Perform resampling to get a smaller, balanced dataset
    X_resampled = []
    y_resampled = []

    # Ensure balanced classes by resampling equal numbers from each class
    samples_per_class = n_samples // len(np.unique(y_train))

    for label in np.unique(y_train):
        X_class = X_train[y_train == label]

        # If we have more samples than needed, downsample
        if X_class.shape[0] > samples_per_class:
            X_resampled_class, y_resampled_class = resample(
                X_class,
                np.full(X_class.shape[0], label),
                n_samples=samples_per_class,
                random_state=seed
            )
        # If we have fewer samples than needed, upsample
        else:
            X_resampled_class, y_resampled_class = resample(
                X_class,
                np.full(X_class.shape[0], label),
                n_samples=samples_per_class,
                random_state=seed,
                replace=True
            )

        X_resampled.append(X_resampled_class)
        y_resampled.append(y_resampled_class)

    # Combine the resampled data
    X_resampled_original = np.vstack(X_resampled)
    y_resampled = np.concatenate(y_resampled)

    print(f"Resampled dataset shape: {X_resampled_original.shape}")
    print(f"Samples per class: {samples_per_class}")

    # Visualize more samples at different noise levels
    print("Generating visualizations of digit samples at different noise levels...")
    visualize_noise_samples(X_resampled_original, y_resampled, noise_levels, n_samples=20, n_rows=4)

    # Generate decision boundary visualizations
    print("\nGenerating decision boundary visualizations...")
    visualize_decision_boundaries_fixed(
        X_resampled_original,
        y_resampled,
        noise_levels=[0, 0.5, 1.0],  # Subset of noise levels for clearer visualization
        tau_values=[1e-6, 1e-2, 1e-1],  # Subset of tau values
        seed=seed
    )

    # Create noisy datasets for each noise level
    X_resampled_noisy_dict = {}
    X_test_noisy_dict = {}

    for noise_level in noise_levels:
        X_resampled_noisy_dict[noise_level] = add_gaussian_noise(X_resampled_original.copy(), std=noise_level)
        X_test_noisy_dict[noise_level] = add_gaussian_noise(X_test.copy(), std=noise_level)

    # Run grid search over noise levels and regularization parameters
    for noise_level in noise_levels:
        print(f"\n=============== Testing noise level: {noise_level} ===============\n")
        X_resampled_noisy = X_resampled_noisy_dict[noise_level]
        X_test_noisy = X_test_noisy_dict[noise_level]

        # Display example images at this noise level (increased to 10 samples)
        plt.figure(figsize=(20, 8))
        plt.suptitle(f'Original vs Noisy Images (Noise Level = {noise_level})', fontsize=14)

        # Display 10 original images on top row
        for i in range(10):
            plt.subplot(2, 10, i + 1)
            plt.imshow(X_resampled_original[i].reshape(8, 8), cmap='gray')
            plt.title(f'Original: {y_resampled[i]}')
            plt.axis('off')

        # Display 10 noisy images on bottom row
        for i in range(10):
            plt.subplot(2, 10, i + 11)
            plt.imshow(X_resampled_noisy[i].reshape(8, 8), cmap='gray')
            plt.title(f'Noisy: {y_resampled[i]}')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'original_vs_noisy_images_{noise_level}.png')
        plt.close()

        # Dictionary to store matrix condition info for this noise level
        matrix_conditions = {}

        for tau in tau_values:
            print(f"\n----- Testing regularization parameter tau: {tau} -----\n")

            try:
                # Create LDA instance with current tau value
                lda = LDA(n_components=n_components, tau=tau)

                # Fit LDA model on noisy data
                lda.fit(X_resampled_noisy, y_resampled)

                # Transform training and test data
                X_train_lda = lda.transform(X_resampled_noisy)
                X_test_lda = lda.transform(X_test_noisy)

                # Train KNN classifier on transformed data
                classifier = KNeighborsClassifier(n_neighbors=5)
                classifier.fit(X_train_lda, y_resampled)

                # Predict and calculate accuracy
                y_pred = classifier.predict(X_test_lda)
                accuracy = accuracy_score(y_test, y_pred)

                # Store all relevant information
                result = {
                    'noise_level': noise_level,
                    'tau': tau,
                    'accuracy': accuracy,
                    'eigenvalues': [abs(val) for val in lda.eigenvalues],
                    'largest_eigenvalue': max([abs(val) for val in lda.eigenvalues]),
                    'smallest_eigenvalue': min([abs(val) for val in lda.eigenvalues]),
                    'eigenvalue_ratio': max([abs(val) for val in lda.eigenvalues]) / max(
                        min([abs(val) for val in lda.eigenvalues]), 1e-12)
                }

                results_data.append(result)
                print(f"Accuracy with noise level {noise_level}, tau {tau}: {accuracy * 100:.2f}%")

                # Generate 2D visualization for selected tau values (sample a few)
                if n_components >= 2 and tau in [tau_values[0], tau_values[len(tau_values) // 2], tau_values[-1]]:
                    plt.figure(figsize=(12, 10))
                    plt.suptitle(f'LDA 2D Projection (Noise Level = {noise_level}, τ = {tau})', fontsize=14)

                    for label in np.unique(y_resampled):
                        idx = y_resampled == label
                        plt.scatter(X_train_lda[idx, 0], X_train_lda[idx, 1], alpha=0.7, label=f'Digit {label}')

                    plt.xlabel('Component 1', fontsize=12)
                    plt.ylabel('Component 2', fontsize=12)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f'lda_2d_projection_noise_{noise_level}_tau_{tau}.png')
                    plt.close()

            except Exception as e:
                print(f"Error with noise level {noise_level}, tau {tau}: {str(e)}")
                # Add error entry to results
                results_data.append({
                    'noise_level': noise_level,
                    'tau': tau,
                    'accuracy': 0,
                    'error': str(e)
                })

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('lda_noise_regularization_results.csv', index=False)

    # Create heatmap of accuracy vs noise and regularization
    try:
        # Filter out error entries
        accuracy_df = results_df[results_df['accuracy'] > 0].copy()

        # Pivot the data for heatmap
        heatmap_data = accuracy_df.pivot_table(
            index='noise_level',
            columns='tau',
            values='accuracy'
        )

        # Plot heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data * 100, annot=True, fmt='.1f', cmap='viridis',
                    linewidths=.5, cbar_kws={'label': 'Accuracy (%)'})
        plt.title('Accuracy by Noise Level and Regularization Parameter (τ)', fontsize=16)
        plt.xlabel('Regularization Parameter (τ)', fontsize=12)
        plt.ylabel('Noise Level (σ)', fontsize=12)
        plt.tight_layout()
        plt.savefig('accuracy_noise_reg_heatmap.png')
        plt.close()

        # Find optimal regularization for each noise level
        optimal_reg = accuracy_df.loc[accuracy_df.groupby('noise_level')['accuracy'].idxmax()]
        print("\nOptimal regularization parameters for each noise level:")
        print(optimal_reg[['noise_level', 'tau', 'accuracy']])

        # Plot optimal regularization vs noise level
        plt.figure(figsize=(10, 6))
        plt.plot(optimal_reg['noise_level'], optimal_reg['tau'], 'o-', linewidth=2)
        plt.xlabel('Noise Level (σ)', fontsize=12)
        plt.ylabel('Optimal Regularization Parameter (τ)', fontsize=12)
        plt.title('Optimal Regularization vs. Noise Level', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for regularization parameter
        plt.tight_layout()
        plt.savefig('optimal_reg_vs_noise.png')
        plt.close()

        # Plot best accuracy vs noise level
        plt.figure(figsize=(10, 6))
        plt.plot(optimal_reg['noise_level'], optimal_reg['accuracy'] * 100, 'o-', linewidth=2)
        plt.xlabel('Noise Level (σ)', fontsize=12)
        plt.ylabel('Best Accuracy (%)', fontsize=12)
        plt.title('Best Accuracy vs. Noise Level (using optimal regularization)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('best_accuracy_vs_noise.png')
        plt.close()

        # Plot eigenvalue ratio for different regularization values at each noise level
        plt.figure(figsize=(12, 8))
        for noise in noise_levels:
            noise_data = accuracy_df[accuracy_df['noise_level'] == noise]
            if not noise_data.empty:
                plt.plot(noise_data['tau'], noise_data['eigenvalue_ratio'], 'o-',
                         linewidth=2, label=f'Noise = {noise}')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Regularization Parameter (τ)', fontsize=12)
        plt.ylabel('Eigenvalue Ratio (Condition Number)', fontsize=12)
        plt.title('Effect of Regularization on Eigenvalue Ratio at Different Noise Levels', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('eigenvalue_ratio_vs_regularization.png')
        plt.close()

    except Exception as e:
        print(f"Error generating visualization: {str(e)}")

    return results_df


# Function to evaluate optimal components
def evaluate_optimal_components(X_train, y_train, X_test, y_test,
                                noise_levels=[0, 0.5, 1.0],
                                tau_values=[1e-6, 1e-2, 1e-1],
                                max_components=9,  # MNIST has 10 classes, so max is 9
                                seed=42):
    """
    Evaluate the optimal number of LDA components for different noise levels.

    Parameters:
    -----------
    X_train : array-like
        Training data
    y_train : array-like
        Training labels
    X_test : array-like
        Test data
    y_test : array-like
        Test labels
    noise_levels : list, default=[0, 0.5, 1.0]
        Noise levels to test
    tau_values : list, default=[1e-6, 1e-2, 1e-1]
        Regularization parameters to test
    max_components : int, default=9
        Maximum number of components to test (for MNIST, max is classes-1 = 9)
    seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing results
    """
    np.random.seed(seed)

    # List to store results
    results = []

    # Create noisy datasets for each noise level
    X_train_noisy_dict = {}
    X_test_noisy_dict = {}

    for noise_level in noise_levels:
        X_train_noisy_dict[noise_level] = add_gaussian_noise(X_train.copy(), std=noise_level)
        X_test_noisy_dict[noise_level] = add_gaussian_noise(X_test.copy(), std=noise_level)

    # Iterate through all combinations
    for noise_level in noise_levels:
        X_train_noisy = X_train_noisy_dict[noise_level]
        X_test_noisy = X_test_noisy_dict[noise_level]

        print(f"\nEvaluating noise level: {noise_level}")

        for tau in tau_values:
            print(f"  Testing regularization parameter τ: {tau}")

            for n_components in range(1, max_components + 1):
                print(f"    Testing {n_components} components...")

                try:
                    # Create and fit LDA
                    lda = LDA(n_components=n_components, tau=tau)
                    lda.fit(X_train_noisy, y_train)

                    # Transform data
                    X_train_lda = lda.transform(X_train_noisy)
                    X_test_lda = lda.transform(X_test_noisy)

                    # Train classifier
                    classifier = KNeighborsClassifier(n_neighbors=5)
                    classifier.fit(X_train_lda, y_train)

                    # Calculate accuracy
                    y_pred = classifier.predict(X_test_lda)
                    accuracy = accuracy_score(y_test, y_pred)

                    # Store results
                    results.append({
                        'noise_level': noise_level,
                        'tau': tau,
                        'n_components': n_components,
                        'accuracy': accuracy,
                        'eigenvalue_ratio': max([abs(val) for val in lda.eigenvalues]) /
                                            max(min([abs(val) for val in lda.eigenvalues]), 1e-12)
                    })

                except Exception as e:
                    print(f"    Error: {str(e)}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('lda_component_evaluation.csv', index=False)

    # Create visualizations
    try:
        # Plot accuracy vs components for each noise level and tau
        plt.figure(figsize=(15, 10))

        for noise_level in noise_levels:
            for tau in tau_values:
                subset = results_df[(results_df['noise_level'] == noise_level) &
                                    (results_df['tau'] == tau)]
                if not subset.empty:
                    plt.plot(subset['n_components'], subset['accuracy'] * 100, 'o-',
                             linewidth=2, label=f'Noise={noise_level}, τ={tau}')

        plt.xlabel('Number of LDA Components', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Classification Accuracy vs. Number of LDA Components', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('accuracy_vs_components.png')
        plt.close()

        # Find optimal components for each combination
        best_components = results_df.loc[results_df.groupby(['noise_level', 'tau'])['accuracy'].idxmax()]
        print("\nOptimal number of components for each combination:")
        print(best_components[['noise_level', 'tau', 'n_components', 'accuracy']])

        # Plot optimal components vs noise level for each tau
        plt.figure(figsize=(10, 6))

        for tau in tau_values:
            subset = best_components[best_components['tau'] == tau]
            if not subset.empty:
                plt.plot(subset['noise_level'], subset['n_components'], 'o-',
                         linewidth=2, label=f'τ={tau}')

        plt.xlabel('Noise Level (σ)', fontsize=12)
        plt.ylabel('Optimal Number of Components', fontsize=12)
        plt.title('Optimal LDA Components vs. Noise Level', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('optimal_components_vs_noise.png')
        plt.close()

        # Create heatmap of accuracy by components and noise level (for best tau)
        for tau in tau_values:
            subset = results_df[results_df['tau'] == tau]

            if not subset.empty:
                pivot_table = subset.pivot_table(
                    index='noise_level',
                    columns='n_components',
                    values='accuracy'
                )

                plt.figure(figsize=(14, 8))
                sns.heatmap(pivot_table * 100, annot=True, fmt='.1f', cmap='viridis',
                            linewidths=.5, cbar_kws={'label': 'Accuracy (%)'})
                plt.title(f'Accuracy by Noise Level and Number of Components (τ={tau})', fontsize=16)
                plt.xlabel('Number of Components', fontsize=12)
                plt.ylabel('Noise Level (σ)', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'components_heatmap_tau_{tau}.png')
                plt.close()

    except Exception as e:
        print(f"Error generating visualization: {str(e)}")

    return results_df


if __name__ == "__main__":
    # Run the main experiment
    results = run_noise_reg_experiment(n_samples=2000, n_components=4, seed=42)

    print("\nExperiment complete. Results saved to 'lda_noise_regularization_results.csv'")

    # Now run the component evaluation
    print("\nEvaluating optimal number of components...")

    # Load and prepare data
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Perform resampling to get a smaller dataset
    X_resampled = []
    y_resampled = []
    samples_per_class = 100  # 100 samples per class, 1000 total

    for label in np.unique(y_train):
        X_class = X_train[y_train == label]

        # If we have more samples than needed, downsample
        if X_class.shape[0] > samples_per_class:
            X_resampled_class, y_resampled_class = resample(
                X_class,
                np.full(X_class.shape[0], label),
                n_samples=samples_per_class,
                random_state=42
            )
        # If we have fewer samples than needed, upsample
        else:
            X_resampled_class, y_resampled_class = resample(
                X_class,
                np.full(X_class.shape[0], label),
                n_samples=samples_per_class,
                random_state=42,
                replace=True
            )

        X_resampled.append(X_resampled_class)
        y_resampled.append(y_resampled_class)

        # Combine the resampled data
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.concatenate(y_resampled)

    # Run the component evaluation
    component_results = evaluate_optimal_components(
        X_resampled, y_resampled, X_test, y_test,
        noise_levels=[0, 0.2, 0.5, 1.0, 2.0],
        tau_values=[1e-6, 1e-2, 1e-1],
        max_components=4
    )

    print("\nComponent evaluation complete. Results saved to 'lda_component_evaluation.csv'")