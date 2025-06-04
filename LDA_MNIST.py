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



# Main script with resampling functionality
def run_lda_mnist(n_components=9, seed=42):
    # Set a fixed random seed for reproducibility
    np.random.seed(seed)

    # Load the full MNIST dataset (28x28 pixels)
    print("Loading full MNIST dataset (28x28)...")

    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')


    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # First, split the original data to ensure consistent test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    # Perform sample
    X_resampled = X_train
    y_resampled = y_train

    # Visualize some samples from the resampled dataset (now 28x28)
    plt.figure(figsize=(15, 3))
    plt.suptitle('Resampled Images', fontsize=14)
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_resampled[i].reshape(28, 28), cmap='gray')  # Changed reshape to 28x28
        plt.title(f'Digit: {y_resampled[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('resampled_images_28x28.png')  # Updated filename

    plt.show()

    print('Distribution per each digit:')
    print(np.bincount(y_resampled))  # shows distribution per digit

    # Apply LDA with regularization
    print(f"Using {n_components} components for LDA")
    lda = LDA(n_components=n_components)

    try:
        # Fit LDA model
        lda.fit(X_resampled, y_resampled)

        # Print and save eigenvalues
        print("\nEigenvalues:")
        eigenvalue_data = []
        for i, val in enumerate(lda.eigenvalues):
            if hasattr(val, 'real'):  # Check if complex
                print(f"Eigenvalue {i + 1}: {val.real:.6f} + {val.imag:.6f}j (magnitude: {abs(val):.6f})")
                eigenvalue_data.append({
                    'Component': i + 1,
                    'Real': val.real,
                    'Imaginary': val.imag,
                    'Magnitude': abs(val)
                })
            else:
                print(f"Eigenvalue {i + 1}: {val:.6f}")
                eigenvalue_data.append({
                    'Component': i + 1,
                    'Real': val,
                    'Imaginary': 0,
                    'Magnitude': abs(val)
                })

        # Calculate variance explained
        eigenvalue_sum = np.sum(np.abs(lda.eigenvalues))
        for i, val in enumerate(eigenvalue_data):
            val['Variance_Explained'] = val['Magnitude'] / eigenvalue_sum * 100
            print(f"Component {i + 1} explains {val['Variance_Explained']:.2f}% of variance")

        # Visualize eigenvalues
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, len(lda.eigenvalues) + 1), [abs(val) for val in lda.eigenvalues])
        plt.xlabel('Component Number', fontsize=12)
        plt.ylabel('Eigenvalue Magnitude', fontsize=12)
        plt.title('Eigenvalues of LDA Components', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig('eigenvalues_plot_28x28.png')  # Updated filename
        plt.show()

        # Transform both training and test data
        X_train_lda = lda.transform(X_resampled)
        X_test_lda = lda.transform(X_test)

        print(f"LDA-transformed training data shape: {X_train_lda.shape}")
        print(f"LDA-transformed test data shape: {X_test_lda.shape}")

        sample_size = 3000
        sample_idx = np.random.choice(len(X_train_lda), sample_size, replace=False)
        X_lda_sampled = X_train_lda[sample_idx]
        y_sampled = y_resampled[sample_idx]

        knn_results = []

        k_values = [1, 3, 5, 7, 9, 11, 15, 21]



        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_lda, y_train)
            y_pred = knn.predict(X_test_lda)
            acc = accuracy_score(y_test, y_pred)
            knn_results.append({'k': k, 'accuracy': acc})
            print(f"k={k}, Accuracy={acc:.4f}")

        # Save to CSV
        knn_df = pd.DataFrame(knn_results)
        knn_df.to_csv("knn_k_vs_accuracy.csv", index=False)



        plt.figure(figsize=(8, 6))
        plt.plot(k_values, [entry['accuracy'] for entry in knn_results], marker='o')

        plt.title("KNN Accuracy vs. Number of Neighbors")
        plt.xlabel("k (Number of Neighbors)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig("knn_accuracy_vs_k.png")
        plt.show()

        # Train a classifier on the LDA-transformed data
        print("\nTraining a KNN classifier on LDA-reduced data...")
        classifier = KNeighborsClassifier(n_neighbors=7)
        classifier.fit(X_train_lda, y_resampled)

        # Make predictions
        y_pred = classifier.predict(X_test_lda)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy * 100:.2f}%")

        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        visualize_specific_misclassifications(y_test, y_pred, X_test)



        # Parse report into dictionary
        report = classification_report(y_test, y_pred, output_dict=True)
        digits = [str(i) for i in range(10)]
        f1_scores = [report[d]['f1-score'] for d in digits]

        # Plot F1-score per digit
        plt.figure(figsize=(8, 5))
        sns.barplot(x=digits, y=f1_scores, hue=digits, palette='Blues', legend=False)

        plt.title("F1-Score per Digit Class")
        plt.xlabel("Digit")
        plt.ylabel("F1-Score")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.savefig("f1_score_per_digit.png")
        plt.show()

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix_28x28.png')  # Updated filename
        plt.show()

        # === Per-Class Accuracy ===
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        digits = [str(i) for i in range(10)]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=digits, y=class_accuracy, hue=digits, palette="viridis", legend=False)
        plt.title("Per-Class Accuracy (Clean LDA on MNIST)")
        plt.xlabel("Digit")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.savefig("per_class_accuracy_clean.png")
        plt.show()



        # === Misclassifications per class ===
        misclassified_per_class = cm.sum(axis=1) - cm.diagonal()
        total_misclassified = misclassified_per_class.sum()

        # Print table
        print("\nMisclassifications per digit:")
        for i, count in enumerate(misclassified_per_class):
            print(f"Digit {i}: {int(count)} misclassifications")

        print(f"\nTotal Misclassifications: {int(total_misclassified)} out of {len(y_test)} samples")
        print(f"Overall Error Rate: {(total_misclassified / len(y_test)) * 100:.2f}%")

        # Plot bar chart of misclassifications
        digits = [str(i) for i in range(10)]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=digits, y=misclassified_per_class, hue=digits, palette="Reds", legend=False)
        plt.title("Number of Misclassified Samples per Digit")
        plt.xlabel("Digit")
        plt.ylabel("Misclassifications")
        plt.grid(True, alpha=0.3)
        plt.savefig("misclassifications_per_digit.png")
        plt.show()

        df_misclass = pd.DataFrame({
            'Digit': list(range(10)),
            'Misclassified': misclassified_per_class.astype(int)
        })
        df_misclass.to_csv("misclassified_per_class.csv", index=False)

        # Identify the most confused digit pair (off-diagonal max)
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)

        # Find indices of the max confusion
        max_idx = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
        confused_pair = (max_idx[0], max_idx[1])
        print(f"Most confused digits: {confused_pair[0]} vs {confused_pair[1]} "
              f"(Count: {cm[max_idx]})")

        # Visualize examples of the confusion
        conf_a, conf_b = confused_pair
        examples = np.where((y_test == conf_a) & (y_pred == conf_b))[0]

        if len(examples) > 0:
            plt.figure(figsize=(10, 2))
            plt.suptitle(f"Examples of Digit {conf_a} misclassified as {conf_b}", fontsize=14)

            for i, idx in enumerate(examples[:5]):
                plt.subplot(1, 5, i + 1)
                plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'misclassified_{conf_a}_as_{conf_b}.png')
            plt.show()
        else:
            print(f"No examples found for digit {conf_a} misclassified as {conf_b}")

        # Plot a few examples of correct and incorrect classifications
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Examples of Classifications (Green: Correct, Red: Incorrect)', fontsize=16)

        # Find some correct and incorrect predictions
        correct_idx = np.where(y_test == y_pred)[0]
        incorrect_idx = np.where(y_test != y_pred)[0]

        # Display 5 correct predictions
        for i in range(5):
            if i < len(correct_idx):
                idx = correct_idx[i]
                axes[0, i].imshow(X_test[idx].reshape(28, 28), cmap='gray')  # Changed reshape to 28x28
                axes[0, i].set_title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}', color='green')
                axes[0, i].axis('off')

        # Display 5 incorrect predictions
        for i in range(5):
            if i < len(incorrect_idx):
                idx = incorrect_idx[i]
                axes[1, i].imshow(X_test[idx].reshape(28, 28), cmap='gray')  # Changed reshape to 28x28
                axes[1, i].set_title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}', color='red')
                axes[1, i].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('classification_examples_28x28.png')  # Updated filename
        plt.show()

        # Normalized Confusion Matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("normalized_confusion_matrix.png")
        plt.show()
        # Create the plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')

        # Custom title showing tau and k values
        plt.title(f"Normalized Confusion Matrix (tau={tau:.1e}, k={k})")

        # Labels for axes
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Save the plot with a filename that includes tau and k
        plt.savefig(f"normalized_confusion_matrix_tau_{tau:.1e}_k_{k}.png")

        # Show the plot
        plt.show()
        # Flatten off-diagonal values from confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        cm_no_diag = conf_matrix.copy()
        np.fill_diagonal(cm_no_diag, 0)

        # Get all off-diagonal pairs with their confusion counts
        confusion_pairs = []
        for i in range(10):
            for j in range(10):
                if i != j:
                    confusion_pairs.append(((i, j), cm_no_diag[i, j]))

        # Sort by highest confusion counts
        top_confusions = sorted(confusion_pairs, key=lambda x: x[1], reverse=True)[:5]

        # Bar plot for top 5 confused digit pairs
        labels = [f"{i}→{j}" for (i, j), _ in top_confusions]
        counts = [count for _, count in top_confusions]

        # Define a grayscale color palette (from black to light grey)
        grey_palette = ["#000000", "#333333", "#666666", "#999999", "#CCCCCC"]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=labels, y=counts, hue=labels, palette=grey_palette, dodge=False, legend=False)
        plt.title("Top 5 Most Confused Digit Pairs")
        plt.ylabel("Number of Misclassifications")
        plt.xlabel("Digit Pair (True → Predicted)")
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig("top_5_confused_pairs.png")
        plt.show()

        # If 2 or 3 components, we can visualize the data in 2D or 3D
        if n_components >= 2:
            colors = sns.color_palette("tab10", 10)

            # Optional: Subsample for clarity
            sample_size = 4000
            sample_idx = np.random.choice(len(X_train_lda), sample_size, replace=False)

            X_lda_sampled = X_train_lda[sample_idx]
            y_sampled = y_resampled[sample_idx]

            plt.figure(figsize=(12, 10))
            for label in np.unique(y_sampled):
                idx = y_sampled == label
                plt.scatter(X_lda_sampled[idx, 0], X_lda_sampled[idx, 1],
                            label=f'Digit {label}',
                            color=colors[label],
                            alpha=0.5, edgecolor='k', linewidth=0.2, s=20)

            # Text centroids
            for label in np.unique(y_sampled):
                idx = y_sampled == label
                x_mean = X_lda_sampled[idx, 0].mean()
                y_mean = X_lda_sampled[idx, 1].mean()
                plt.text(x_mean, y_mean, str(label), fontsize=12, weight='bold', color='black')

            plt.xlabel('Component 1', fontsize=12)
            plt.ylabel('Component 2', fontsize=12)
            plt.title('Improved LDA 2D Projection of MNIST Digits', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('lda_2d_projection_improved.png')
            plt.show()




            # Use consistent colors for digit labels
            colors = sns.color_palette("tab10", 10)

            # Optional: Subsample for clarity
            sample_size = 3000
            sample_idx = np.random.choice(len(X_train_lda), sample_size, replace=False)
            X_lda_sampled = X_train_lda[sample_idx]
            y_sampled = y_resampled[sample_idx]

            # Prepare DataFrame for violin plots
            df_lda = pd.DataFrame(X_lda_sampled, columns=[f"Comp {i + 1}" for i in range(X_lda_sampled.shape[1])])
            df_lda["Digit"] = y_sampled.astype(str)

            # Variance explained (grab from eigenvalue_data)
            component_variances = [val['Variance_Explained'] for val in eigenvalue_data]

            plt.figure(figsize=(12, 10))
            for label in np.unique(y_sampled):
                idx = y_sampled == label
                plt.scatter(X_lda_sampled[idx, 0], X_lda_sampled[idx, 2],  # component 1 vs 3
                            label=f'Digit {label}',
                            color=colors[label],
                            alpha=0.5, edgecolor='k', linewidth=0.2, s=20)

            # Add centroid text labels
            for label in np.unique(y_sampled):
                idx = y_sampled == label
                x_mean = X_lda_sampled[idx, 0].mean()
                y_mean = X_lda_sampled[idx, 2].mean()
                plt.text(x_mean, y_mean, str(label), fontsize=12, weight='bold', color='black')

            # Add axis labels with variance %
            plt.xlabel(f'Component 1 ({component_variances[0]:.2f}% variance)', fontsize=12)
            plt.ylabel(f'Component 3 ({component_variances[2]:.2f}% variance)', fontsize=12)

            plt.title('LDA Projection of MNIST Digits (Components 1 vs. 3)', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Export high-quality version for thesis
            plt.savefig('lda_2d_projection_c1_c3.pdf', dpi=300)  # PDF for vector quality
            plt.savefig('lda_2d_projection_c1_c3.png', dpi=300)  # PNG fallback
            plt.show()

            # 3D plot if we have at least 3 components
            if n_components >= 3:
                from mpl_toolkits.mplot3d import Axes3D

                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')

                for label in np.unique(y_resampled):
                    idx = y_resampled == label
                    ax.scatter(X_train_lda[idx, 0], X_train_lda[idx, 1], X_train_lda[idx, 2],
                               alpha=0.7, label=f'Digit {label}')

                ax.set_xlabel('Component 1', fontsize=12)
                ax.set_ylabel('Component 2', fontsize=12)
                ax.set_zlabel('Component 3', fontsize=12)
                ax.set_title('LDA 3D Projection of MNIST Digits (28x28)', fontsize=14)  # Updated title
                plt.legend()
                plt.savefig('lda_3d_projection_28x28.png')  # Updated filename
                plt.show()



                # -- Violin Plots (Each Component vs Digit) --
                for i in range(9):
                    plt.figure(figsize=(10, 4))
                    sns.violinplot(x="Digit", y=f"Comp {i + 1}", hue="Digit", data=df_lda,
                                   palette="tab10", inner="quart", legend=False)

                    plt.title(f"LDA Component {i + 1} Distribution Across Digits")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"lda_component_{i + 1}_violin.png")
                    plt.show()

        return lda, classifier, X_train_lda, X_test_lda, y_resampled, y_test, y_pred


    except Exception as e:
        print(f"Error during LDA/classification: {str(e)}")
        return None, None, None, None, None, None, None


# Function to visualize specific misclassifications (e.g., 3→5, 8→1)
def visualize_specific_misclassifications(y_test, y_pred, X_test, digit_pairs=[(3, 5), (8, 1), (9, 4), (5, 3)]):
    for true_digit, predicted_digit in digit_pairs:
        examples = np.where((y_test == true_digit) & (y_pred == predicted_digit))[0]

        if len(examples) > 0:
            plt.figure(figsize=(10, 2))
            plt.suptitle(f"Examples of Digit {true_digit} misclassified as {predicted_digit}", fontsize=14)

            for i, idx in enumerate(examples[:5]):  # Show up to 5 examples
                plt.subplot(1, 5, i + 1)
                plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
                plt.title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}')
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(f'misclassified_{true_digit}_as_{predicted_digit}.png')
            plt.show()
        else:
            print(f"No examples found for digit {true_digit} misclassified as {predicted_digit}")


def evaluate_accuracy_over_tau(tau_values, n_components=9, seed=42):
    """
    For each tau, perform LDA + KNN, save eigenvalue plot and normalized confusion matrix.
    """
    from LDA_Function import LDA
    accuracies = []

    # Load data once
    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    for tau in tau_values:
        print(f"\nEvaluating tau = {tau:.1e}")
        lda = LDA(n_components=n_components, tau=tau)
        lda.fit(X_train, y_train)

        # === Plot eigenvalues ===
        eigenvals = np.abs(lda.eigenvalues)
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(eigenvals) + 1), eigenvals)
        plt.xlabel("LDA Component")
        plt.ylabel("Eigenvalue Magnitude")
        plt.title(f"LDA Eigenvalues (tau={tau:.1e})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"eigenvalues_tau_{tau:.0e}.png")
        plt.close()

        # === Transform data ===
        X_train_lda = lda.transform(X_train)
        X_test_lda = lda.transform(X_test)

        # === Train and predict with KNN ===
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X_train_lda, y_train)
        y_pred = knn.predict(X_test_lda)

        # === Accuracy ===
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Accuracy: {acc * 100:.2f}%")

        # === Normalized confusion matrix ===
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=False, fmt=".2f", cmap="Blues")
        plt.title(f"Normalized Confusion Matrix (tau={tau:.1e})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_tau_{tau:.0e}.png")
        plt.close()

    # Summary plot (optional)
    plt.figure(figsize=(8, 5))
    plt.semilogx(tau_values, accuracies, marker='o')
    plt.title("Accuracy vs. Tau")
    plt.xlabel("Tau (log scale)")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.savefig("accuracy_vs_tau_summary.png")
    plt.close()

    # Save results
    results_df = pd.DataFrame({'tau': tau_values, 'accuracy': accuracies})
    results_df.to_csv("tau_vs_accuracy_detailed.csv", index=False)

    return results_df


def load_clean_mnist_split(seed=42):
    """
    Load the raw MNIST dataset and split into training and test sets.
    No LDA, no projection — just raw 784-D images.
    """
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    return train_test_split(X, y, test_size=0.3, random_state=seed)






# Run the full LDA pipeline using 9 components and full training data

if __name__ == "__main__":
    lda, classifier, X_train_lda, X_test_lda, y_train, y_test, y_pred = run_lda_mnist(
        n_components=9,
        seed=42
    )
    # ==== Evaluate accuracy over different tau values ====
    tau_values = [0.0,1e-12 ,1e-10, 1e-8, 1e-6, 1e-4]
    results = evaluate_accuracy_over_tau(tau_values, n_components=9, seed=42)

    print("\nSummary of Accuracy for Each Tau:")
    for idx, row in results.iterrows():
        tau = row["tau"]
        acc = row["accuracy"]
        print(f"Tau = {tau:.0e} → Accuracy = {acc * 100:.2f}%")

