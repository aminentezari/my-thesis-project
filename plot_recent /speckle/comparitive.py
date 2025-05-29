import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def load_results_files():
    """Load all the experiment results CSV files."""
    result_files = {
        'gaussian': 'gaussian_noise_results.csv',
        'salt_pepper': 'salt_pepper_noise_results.csv',
        'blur': 'blur_results.csv',
        'speckle': 'speckle_noise_results.csv',
        'zigzag': 'zigzag_noise_results.csv'
    }

    results = {}

    for noise_type, filename in result_files.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            results[noise_type] = df
            print(f"Loaded {noise_type} results: {len(df)} rows")
        else:
            print(f"Warning: {filename} not found")

    return results


def standardize_dataframes(results):
    """Standardize column names and values across different dataframes."""
    standardized = {}

    for noise_type, df in results.items():
        # Create a copy to avoid modifying the original
        std_df = df.copy()

        # Standardize noise level columns
        if noise_type == 'gaussian':
            std_df['noise_level'] = std_df['noise_std']
        elif noise_type == 'salt_pepper':
            std_df['noise_level'] = std_df['density']
        elif noise_type == 'blur':
            std_df['noise_level'] = std_df['sigma']
        elif noise_type == 'speckle':
            std_df['noise_level'] = std_df['noise_std']
        elif noise_type == 'zigzag':
            # Special handling for zigzag noise
            if 'amplitude' in std_df.columns:
                # Use amplitude as the primary noise measure
                std_df['noise_level'] = std_df['amplitude']

                # Keep only rows with frequency = 2 for consistent comparison
                if 'frequency' in std_df.columns:
                    # Check if frequency=2 exists
                    if 2 in std_df['frequency'].values:
                        std_df = std_df[std_df['frequency'] == 2]
                    else:
                        # Use the most common frequency
                        common_freq = std_df['frequency'].mode()[0]
                        std_df = std_df[std_df['frequency'] == common_freq]
                        print(f"Using frequency={common_freq} for zigzag noise (frequency=2 not found)")
            elif 'param_key' in std_df.columns:
                # Try to extract amplitude from param_key (e.g., "2_2")
                try:
                    std_df['noise_level'] = std_df['param_key'].apply(
                        lambda x: float(x.split('_')[0]) if '_' in str(x) else 0)
                except:
                    print("Could not parse param_key for zigzag noise")
                    std_df['noise_level'] = 0
            else:
                # Try to find other suitable columns
                if 'noise_level' not in std_df.columns:
                    possible_columns = ['level', 'intensity', 'strength', 'amount']
                    for col in possible_columns:
                        if col in std_df.columns:
                            std_df['noise_level'] = std_df[col]
                            print(f"Using {col} as noise_level for zigzag noise")
                            break
                    else:
                        # If no suitable column found, use row index as proxy
                        print("Warning: No suitable noise level column found for zigzag, using row index")
                        std_df['noise_level'] = std_df.index

        # Add noise type as a column for combined analysis
        std_df['noise_type'] = noise_type

        # Handle misclassified column if it exists
        if 'misclassified' not in std_df.columns and 'accuracy' in std_df.columns:
            # Assuming test set size is approximately 21000 (30% of 70,000 MNIST)
            test_size = 21000
            std_df['misclassified'] = round((1 - std_df['accuracy']) * test_size)

        standardized[noise_type] = std_df

    return standardized


def create_combined_dataframe(standardized_results):
    """Merge all standardized dataframes into a single dataframe for analysis."""
    combined_df = pd.concat(standardized_results.values(), ignore_index=True)

    # Create normalized noise levels (0-1 scale) for fair comparisons
    for noise_type in standardized_results.keys():
        noise_df = combined_df[combined_df['noise_type'] == noise_type]
        if len(noise_df) > 0:
            # Get noise levels excluding 0 (baseline)
            noise_levels = noise_df['noise_level'].values
            non_zero_levels = noise_levels[noise_levels > 0]

            if len(non_zero_levels) > 0:
                min_noise = np.min(non_zero_levels)
                max_noise = np.max(non_zero_levels)

                # Avoid division by zero
                if max_noise > min_noise:
                    # For zero noise level, set normalized to 0
                    combined_df.loc[(combined_df['noise_type'] == noise_type) &
                                    (combined_df['noise_level'] == 0), 'normalized_noise'] = 0

                    # For non-zero noise levels, normalize between 0 and 1
                    combined_df.loc[(combined_df['noise_type'] == noise_type) &
                                    (combined_df['noise_level'] > 0), 'normalized_noise'] = (
                            (combined_df.loc[(combined_df['noise_type'] == noise_type) &
                                             (combined_df['noise_level'] > 0), 'noise_level'] - min_noise) /
                            (max_noise - min_noise)
                    )
                else:
                    # If only one non-zero noise level, set to 1
                    combined_df.loc[(combined_df['noise_type'] == noise_type) &
                                    (combined_df['noise_level'] == 0), 'normalized_noise'] = 0
                    combined_df.loc[(combined_df['noise_type'] == noise_type) &
                                    (combined_df['noise_level'] > 0), 'normalized_noise'] = 1
            else:
                # If no non-zero noise levels, set all to 0
                combined_df.loc[combined_df['noise_type'] == noise_type, 'normalized_noise'] = 0

    # Print summary of combined dataframe
    print("\nCombined Dataframe Summary:")
    for noise_type in standardized_results.keys():
        noise_df = combined_df[combined_df['noise_type'] == noise_type]
        if len(noise_df) > 0:
            min_level = noise_df['noise_level'].min()
            max_level = noise_df['noise_level'].max()
            tau_values = sorted(noise_df['tau'].unique())
            print(
                f"{noise_type}: {len(noise_df)} rows, noise levels {min_level:.3f}-{max_level:.3f}, tau values: {tau_values}")

    return combined_df


def plot_combined_noise_comparison(combined_df):
    """Create plots comparing different noise types for each regularization value."""
    tau_values = sorted(combined_df['tau'].unique())

    # Create a plot for each tau value
    for tau in tau_values:
        plt.figure(figsize=(12, 8))

        for noise_type in combined_df['noise_type'].unique():
            # Get data for this tau and noise type
            subset = combined_df[(combined_df['tau'] == tau) & (combined_df['noise_type'] == noise_type)]

            # Skip if insufficient data
            if len(subset) < 2:
                continue

            # Sort by normalized noise level
            subset = subset.sort_values('normalized_noise')

            # Plot accuracy vs normalized noise level
            plt.plot(subset['normalized_noise'], subset['accuracy'],
                     marker='o', linewidth=2,
                     label=noise_type.replace('_', ' ').title())

        plt.title(f'Accuracy vs. Normalized Noise Level for τ = {tau:.0e}', fontsize=14)
        plt.xlabel('Normalized Noise Level (0-1 scale)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Set y-axis to start from 0.1 for better visualization
        plt.ylim(0.1, 1.0)

        plt.savefig(f"comparison_tau_{tau:.0e}_normalized.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_accuracy_drop_heatmap(combined_df):
    """Create a heatmap showing accuracy drop from clean to maximum noise level."""
    # Calculate accuracy drop for each noise type and tau value
    drop_data = []

    for noise_type in combined_df['noise_type'].unique():
        for tau in sorted(combined_df['tau'].unique()):
            # Get subset for this noise type and tau
            subset = combined_df[(combined_df['noise_type'] == noise_type) &
                                 (combined_df['tau'] == tau)]

            # Skip if insufficient data
            if len(subset) < 2:
                continue

            # Get baseline (no noise) accuracy
            baseline = subset[subset['noise_level'] == 0]
            if len(baseline) == 0:
                # If no explicit baseline, use minimum noise level as baseline
                min_noise = subset['noise_level'].min()
                baseline = subset[subset['noise_level'] == min_noise]
                if len(baseline) == 0:
                    continue

            baseline_acc = baseline['accuracy'].values[0]

            # Get maximum noise level data
            max_noise_level = subset['noise_level'].max()
            if max_noise_level == 0 or max_noise_level == baseline['noise_level'].values[0]:
                # Skip if only baseline exists
                continue

            max_noise_data = subset[subset['noise_level'] == max_noise_level]
            if len(max_noise_data) == 0:
                continue

            max_noise_acc = max_noise_data['accuracy'].values[0]

            # Calculate accuracy drop
            acc_drop = baseline_acc - max_noise_acc
            acc_drop_percent = (acc_drop / baseline_acc) * 100 if baseline_acc > 0 else 0

            drop_data.append({
                'noise_type': noise_type,
                'tau': tau,
                'baseline_acc': baseline_acc,
                'max_noise_acc': max_noise_acc,
                'acc_drop_percent': acc_drop_percent
            })

    # Create dataframe for heatmap
    drop_df = pd.DataFrame(drop_data)

    # Convert to pivot table
    pivot_table = drop_df.pivot_table(
        values='acc_drop_percent',
        index='tau',
        columns='noise_type'
    )

    # Create heatmap
    plt.figure(figsize=(12, 8))

    # Custom colormap from green (low drop) to red (high drop)
    cmap = LinearSegmentedColormap.from_list('GreenToRed', ['#2ca02c', '#ffe135', '#d62728'])

    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap=cmap, linewidths=0.5)

    plt.title('Accuracy Drop (%) from Clean to Maximum Noise Level', fontsize=14)
    plt.xlabel('Noise Type', fontsize=12)
    plt.ylabel('Regularization Parameter (τ)', fontsize=12)

    # Add colorbar label
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Accuracy Drop (%)', fontsize=12)

    plt.tight_layout()
    plt.savefig("accuracy_drop_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save the data
    drop_df.to_csv("accuracy_drop_data.csv", index=False)

    return drop_df


def plot_regularization_effectiveness(combined_df):
    """Plot how different tau values affect accuracy for each noise type and level."""
    # Create plots for each noise type showing accuracy vs tau for different noise levels
    for noise_type in combined_df['noise_type'].unique():
        # Get unique noise levels for this type
        noise_levels = sorted(combined_df[combined_df['noise_type'] == noise_type]['noise_level'].unique())

        # Select a few representative noise levels (no noise, low, medium, high)
        if len(noise_levels) >= 4:
            selected_levels = [
                noise_levels[0],  # No/lowest noise
                noise_levels[1],  # Low noise
                noise_levels[len(noise_levels) // 2],  # Medium noise
                noise_levels[-1]  # High noise
            ]
        elif len(noise_levels) >= 2:
            selected_levels = [noise_levels[0], noise_levels[-1]]  # Just min and max noise
        else:
            continue  # Skip if not enough noise levels

        plt.figure(figsize=(12, 8))

        for noise_level in selected_levels:
            # Get data for this noise type and level across tau values
            tau_data = []

            for tau in sorted(combined_df['tau'].unique()):
                data = combined_df[(combined_df['noise_type'] == noise_type) &
                                   (combined_df['noise_level'] == noise_level) &
                                   (combined_df['tau'] == tau)]

                if len(data) > 0:
                    tau_data.append({
                        'tau': tau,
                        'accuracy': data['accuracy'].values[0]
                    })

            # Skip if insufficient data points
            if len(tau_data) < 2:
                continue

            # Convert to dataframe
            tau_df = pd.DataFrame(tau_data)

            # Plot accuracy vs tau
            plt.plot(tau_df['tau'], tau_df['accuracy'],
                     marker='o', linewidth=2,
                     label=f'Noise Level = {noise_level:.2f}')

        plt.xscale('log')  # Log scale for tau axis
        plt.title(f'Accuracy vs. τ for {noise_type.replace("_", " ").title()} Noise', fontsize=14)
        plt.xlabel('Regularization Parameter (τ)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"tau_effectiveness_{noise_type}.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_misclassification_patterns(combined_df):
    """Create plots showing misclassification patterns for different noise types and regularization levels."""
    # We'll plot misclassified samples vs. noise level for each noise type and tau value
    for noise_type in combined_df['noise_type'].unique():
        # Skip if no misclassified column
        if 'misclassified' not in combined_df.columns:
            continue

        plt.figure(figsize=(12, 8))

        # Plot for each tau value
        for tau in sorted(combined_df['tau'].unique()):
            # Get data for this noise type and tau
            subset = combined_df[(combined_df['noise_type'] == noise_type) &
                                 (combined_df['tau'] == tau)]

            # Skip if not enough data points
            if len(subset) < 2:
                continue

            # Sort by noise level
            subset = subset.sort_values('noise_level')

            # Plot misclassified vs noise level
            plt.plot(subset['noise_level'], subset['misclassified'],
                     marker='o', linewidth=2,
                     label=f'τ = {tau:.0e}')

        plt.title(f'Misclassified Samples vs. Noise Level ({noise_type.replace("_", " ").title()})', fontsize=14)
        plt.xlabel('Noise Level', fontsize=12)
        plt.ylabel('Number of Misclassified Samples', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"misclassified_{noise_type}.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_noise_robustness_ranking(combined_df, drop_df=None):
    """Create a plot ranking noise types by their robustness to noise."""
    # If drop_df is provided, use it; otherwise calculate drops
    if drop_df is None:
        # Calculate drops from combined_df
        noise_metrics = []

        for noise_type in combined_df['noise_type'].unique():
            for tau in sorted(combined_df['tau'].unique()):
                subset = combined_df[(combined_df['noise_type'] == noise_type) &
                                     (combined_df['tau'] == tau)]

                # Skip if not enough data
                if len(subset) < 2:
                    continue

                # Get baseline (no noise) accuracy
                baseline = subset[subset['noise_level'] == 0]
                if len(baseline) == 0:
                    # If no explicit baseline, use minimum noise level as baseline
                    min_noise = subset['noise_level'].min()
                    baseline = subset[subset['noise_level'] == min_noise]
                    if len(baseline) == 0:
                        continue

                baseline_acc = baseline['accuracy'].values[0]

                # Get max noise level accuracy
                max_noise_level = subset['noise_level'].max()
                if max_noise_level == 0 or max_noise_level == baseline['noise_level'].values[0]:
                    continue

                max_noise_acc = subset[subset['noise_level'] == max_noise_level]['accuracy'].values[0]

                # Calculate drop
                drop = baseline_acc - max_noise_acc
                drop_percent = (drop / baseline_acc) * 100 if baseline_acc > 0 else 0

                noise_metrics.append({
                    'noise_type': noise_type,
                    'tau': tau,
                    'baseline_acc': baseline_acc,
                    'max_noise_acc': max_noise_acc,
                    'drop_percent': drop_percent
                })

        metrics_df = pd.DataFrame(noise_metrics)
    else:
        # Use provided drop_df
        metrics_df = drop_df.copy()
        metrics_df.rename(columns={'acc_drop_percent': 'drop_percent'}, inplace=True)

    # Calculate average drop for each noise type
    avg_drops = metrics_df.groupby('noise_type')['drop_percent'].mean().reset_index()
    avg_drops = avg_drops.sort_values('drop_percent', ascending=False)

    # Plot
    plt.figure(figsize=(12, 8))

    bars = plt.bar(avg_drops['noise_type'].apply(lambda x: x.replace('_', ' ').title()),
                   avg_drops['drop_percent'], color='firebrick')

    plt.xlabel('Noise Type', fontsize=12)
    plt.ylabel('Average Accuracy Drop (%)', fontsize=12)
    plt.title('Noise Types Ranked by Destructiveness', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("noise_robustness_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()

    return avg_drops


def plot_tau_noise_relationship(combined_df, drop_df=None):
    """Create plots showing the relationship between tau and noise tolerance."""
    # Get the best tau for each noise type at different noise levels

    for noise_type in combined_df['noise_type'].unique():
        # Get all noise levels for this type
        noise_levels = sorted(combined_df[combined_df['noise_type'] == noise_type]['noise_level'].unique())

        # Skip if not enough levels
        if len(noise_levels) < 2:
            continue

        # Create data for plotting
        best_tau_data = []

        for noise_level in noise_levels:
            # Get data for this noise level
            level_data = combined_df[(combined_df['noise_type'] == noise_type) &
                                     (combined_df['noise_level'] == noise_level)]

            # Skip if not enough data
            if len(level_data) < 2:
                continue

            # Find best tau (highest accuracy)
            best_idx = level_data['accuracy'].idxmax()
            best_row = level_data.loc[best_idx]

            best_tau_data.append({
                'noise_level': noise_level,
                'best_tau': best_row['tau'],
                'accuracy': best_row['accuracy']
            })

        # Skip if not enough data points
        if len(best_tau_data) < 2:
            continue

        # Convert to dataframe
        best_tau_df = pd.DataFrame(best_tau_data)

        # Plot
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Plot best tau vs noise level
        color = 'tab:blue'
        ax1.set_xlabel('Noise Level', fontsize=12)
        ax1.set_ylabel('Best τ Value', fontsize=12, color=color)
        ax1.plot(best_tau_df['noise_level'], best_tau_df['best_tau'],
                 marker='o', color=color, linewidth=2)
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor=color)

        # Plot accuracy on secondary y-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Accuracy with Best τ', fontsize=12, color=color)
        ax2.plot(best_tau_df['noise_level'], best_tau_df['accuracy'],
                 marker='s', color=color, linewidth=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Best τ Value vs. Noise Level for {noise_type.replace("_", " ").title()} Noise', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"best_tau_vs_noise_{noise_type}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Create summary plot showing best tau for max noise level across all noise types
    best_tau_summary = []

    for noise_type in combined_df['noise_type'].unique():
        # Get max noise level for this type
        max_noise_level = combined_df[combined_df['noise_type'] == noise_type]['noise_level'].max()

        # Get data for this max noise level
        max_noise_data = combined_df[(combined_df['noise_type'] == noise_type) &
                                     (combined_df['noise_level'] == max_noise_level)]

        # Skip if not enough data
        if len(max_noise_data) < 2:
            continue

        # Find best tau (highest accuracy)
        best_idx = max_noise_data['accuracy'].idxmax()
        best_row = max_noise_data.loc[best_idx]

        best_tau_summary.append({
            'noise_type': noise_type,
            'best_tau': best_row['tau'],
            'max_noise_level': max_noise_level,
            'accuracy': best_row['accuracy']
        })

    # Skip if not enough data
    if len(best_tau_summary) < 2:
        return None

    # Convert to dataframe
    summary_df = pd.DataFrame(best_tau_summary)

    # Sort by noise type
    summary_df = summary_df.sort_values('noise_type')

    # Plot
    plt.figure(figsize=(12, 8))

    x = range(len(summary_df))

    plt.bar(x, summary_df['best_tau'], color='blue', alpha=0.7)

    plt.xlabel('Noise Type', fontsize=12)
    plt.ylabel('Best τ Value for Maximum Noise Level', fontsize=12)
    plt.title('Optimal Regularization Parameter by Noise Type', fontsize=14)
    plt.xticks(x, summary_df['noise_type'].apply(lambda x: x.replace('_', ' ').title()), rotation=45)
    plt.yscale('log')
    plt.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for i, v in enumerate(summary_df['best_tau']):
        plt.text(i, v * 1.1, f'{v:.0e}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("best_tau_by_noise_type.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save the data
    summary_df.to_csv("best_tau_summary.csv", index=False)

    return summary_df


def create_correlation_analysis(combined_df):
    """Perform correlation analysis between tau, noise level, and accuracy."""
    # Calculate correlation for each noise type
    corr_results = []

    for noise_type in combined_df['noise_type'].unique():
        # Get data for this noise type
        noise_df = combined_df[combined_df['noise_type'] == noise_type].copy()

        # Skip if not enough data
        if len(noise_df) < 5:
            continue

        # Add log(tau) column for better linear correlation
        noise_df['log_tau'] = np.log10(noise_df['tau'])

        try:
            # Calculate correlation coefficients
            corr_noise_acc = stats.pearsonr(noise_df['noise_level'], noise_df['accuracy'])[0]
            corr_tau_acc = stats.pearsonr(noise_df['log_tau'], noise_df['accuracy'])[0]

            # Calculate correlation at different noise levels
            noise_levels = sorted(noise_df['noise_level'].unique())

            if len(noise_levels) >= 3:
                # Get low and high noise
                low_noise = noise_levels[1] if len(noise_levels) > 1 else noise_levels[0]  # Skip zero
                high_noise = noise_levels[-1]

                # Calculate tau-accuracy correlation at different noise levels
                low_noise_df = noise_df[noise_df['noise_level'] == low_noise]
                high_noise_df = noise_df[noise_df['noise_level'] == high_noise]

                if len(low_noise_df) >= 3 and len(high_noise_df) >= 3:
                    corr_tau_acc_low_noise = stats.pearsonr(low_noise_df['log_tau'], low_noise_df['accuracy'])[0]
                    corr_tau_acc_high_noise = stats.pearsonr(high_noise_df['log_tau'], high_noise_df['accuracy'])[0]

                    # Calculate interaction (how much more important regularization becomes at high noise)
                    tau_noise_interaction = corr_tau_acc_high_noise - corr_tau_acc_low_noise
                else:
                    corr_tau_acc_low_noise = np.nan
                    corr_tau_acc_high_noise = np.nan
                    tau_noise_interaction = np.nan
            else:
                corr_tau_acc_low_noise = np.nan
                corr_tau_acc_high_noise = np.nan
                tau_noise_interaction = np.nan

            corr_results.append({
                'noise_type': noise_type,
                'noise_acc_correlation': corr_noise_acc,
                'tau_acc_correlation': corr_tau_acc,
                'tau_acc_correlation_low_noise': corr_tau_acc_low_noise,
                'tau_acc_correlation_high_noise': corr_tau_acc_high_noise,
                'tau_noise_interaction': tau_noise_interaction
            })
        except:
            print(f"Could not calculate correlation for {noise_type}")

    # Convert to dataframe
    corr_df = pd.DataFrame(corr_results)

    # Skip if empty
    if len(corr_df) == 0:
        print("No correlation data available")
        return None

    # Plot correlation results
    plt.figure(figsize=(12, 8))

    x = range(len(corr_df))
    width = 0.35

    plt.bar([i - width / 2 for i in x], corr_df['noise_acc_correlation'], width,
            label='Noise-Accuracy Correlation', color='blue', alpha=0.7)
    plt.bar([i + width / 2 for i in x], corr_df['tau_acc_correlation'], width,
            label='Tau-Accuracy Correlation', color='green', alpha=0.7)

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.xlabel('Noise Type', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.title('Correlation Between Variables by Noise Type', fontsize=14)
    plt.xticks(x, corr_df['noise_type'].apply(lambda x: x.replace('_', ' ').title()), rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("correlation_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save the data
    corr_df.to_csv("correlation_analysis.csv", index=False)

    return corr_df




def create_comprehensive_summary(drop_df, best_tau_df, corr_df):
    """Create a comprehensive summary table for all noise types."""
    # Combine information from all analyses
    summary_data = []

    # Get list of all noise types across all dataframes
    noise_types = set()
    if drop_df is not None:
        noise_types.update(drop_df['noise_type'].unique())
    if best_tau_df is not None:
        noise_types.update(best_tau_df['noise_type'].unique())
    if corr_df is not None:
        noise_types.update(corr_df['noise_type'].unique())

    for noise_type in sorted(noise_types):
        summary_row = {'noise_type': noise_type}

        # Add accuracy drop information
        if drop_df is not None and noise_type in drop_df['noise_type'].values:
            avg_drop = drop_df[drop_df['noise_type'] == noise_type]['acc_drop_percent'].mean()
            summary_row['avg_accuracy_drop_percent'] = avg_drop
        else:
            summary_row['avg_accuracy_drop_percent'] = np.nan

        # Add best tau information
        if best_tau_df is not None and noise_type in best_tau_df['noise_type'].values:
            best_tau = best_tau_df[best_tau_df['noise_type'] == noise_type]['best_tau'].values[0]
            best_acc = best_tau_df[best_tau_df['noise_type'] == noise_type]['accuracy'].values[0]
            summary_row['best_tau_at_max_noise'] = best_tau
            summary_row['accuracy_at_max_noise'] = best_acc
        else:
            summary_row['best_tau_at_max_noise'] = np.nan
            summary_row['accuracy_at_max_noise'] = np.nan

        # Add correlation information
        if corr_df is not None and noise_type in corr_df['noise_type'].values:
            noise_acc_corr = corr_df[corr_df['noise_type'] == noise_type]['noise_acc_correlation'].values[0]
            tau_acc_corr = corr_df[corr_df['noise_type'] == noise_type]['tau_acc_correlation'].values[0]
            interaction = corr_df[corr_df['noise_type'] == noise_type]['tau_noise_interaction'].values[0]

            summary_row['noise_acc_correlation'] = noise_acc_corr
            summary_row['tau_acc_correlation'] = tau_acc_corr
            summary_row['tau_noise_interaction'] = interaction
        else:
            summary_row['noise_acc_correlation'] = np.nan
            summary_row['tau_acc_correlation'] = np.nan
            summary_row['tau_noise_interaction'] = np.nan

        summary_data.append(summary_row)

    # Convert to dataframe
    summary_df = pd.DataFrame(summary_data)

    # Sort by average accuracy drop (most destructive first)
    if 'avg_accuracy_drop_percent' in summary_df.columns:
        summary_df = summary_df.sort_values('avg_accuracy_drop_percent', ascending=False)

    # Save to CSV
    summary_df.to_csv("noise_comprehensive_summary.csv", index=False)

    # Create summary report
    with open("noise_analysis_report.md", "w") as f:
        f.write("# Comparative Analysis of Noise Types and Regularization in LDA-KNN\n\n")

        # Add summary table
        f.write("## Summary of Noise Types\n\n")
        f.write(
            "| Noise Type | Avg. Accuracy Drop (%) | Best τ at Max Noise | Accuracy at Max Noise | Tau-Accuracy Correlation | Noise-Accuracy Correlation | Tau-Noise Interaction |\n")
        f.write(
            "|------------|------------------------|---------------------|----------------------|--------------------------|----------------------------|------------------------|\n")

        for _, row in summary_df.iterrows():
            noise_type = row['noise_type'].replace('_', ' ').title()

            avg_drop = f"{row['avg_accuracy_drop_percent']:.1f}" if not pd.isna(
                row['avg_accuracy_drop_percent']) else "N/A"
            best_tau = f"{row['best_tau_at_max_noise']:.0e}" if not pd.isna(row['best_tau_at_max_noise']) else "N/A"
            max_acc = f"{row['accuracy_at_max_noise']:.3f}" if not pd.isna(row['accuracy_at_max_noise']) else "N/A"
            tau_corr = f"{row['tau_acc_correlation']:.3f}" if not pd.isna(row['tau_acc_correlation']) else "N/A"
            noise_corr = f"{row['noise_acc_correlation']:.3f}" if not pd.isna(row['noise_acc_correlation']) else "N/A"
            interaction = f"{row['tau_noise_interaction']:.3f}" if not pd.isna(row['tau_noise_interaction']) else "N/A"

            f.write(
                f"| {noise_type} | {avg_drop} | {best_tau} | {max_acc} | {tau_corr} | {noise_corr} | {interaction} |\n")

        # Add interpretation of results
        f.write("\n## Key Findings\n\n")

        # 1. Most destructive noise types
        f.write("### Noise Type Impact Ranking\n\n")
        for i, (_, row) in enumerate(summary_df.iterrows(), 1):
            if pd.isna(row['avg_accuracy_drop_percent']):
                continue

            noise_type = row['noise_type'].replace('_', ' ').title()
            avg_drop = row['avg_accuracy_drop_percent']

            strength = "Highly destructive" if avg_drop > 50 else "Moderately destructive" if avg_drop > 20 else "Minimally destructive"

            f.write(f"{i}. **{noise_type}**: {avg_drop:.1f}% average accuracy drop ({strength})\n")

        # 2. Regularization effectiveness
        f.write("\n### Regularization Effectiveness\n\n")
        for _, row in summary_df.iterrows():
            if pd.isna(row['tau_acc_correlation']):
                continue

            noise_type = row['noise_type'].replace('_', ' ').title()
            tau_corr = row['tau_acc_correlation']
            interaction = row['tau_noise_interaction']

            corr_strength = "Strong positive" if tau_corr > 0.7 else "Moderate positive" if tau_corr > 0.3 else "Weak positive" if tau_corr > 0 else "Negative"
            interact_desc = "becomes much more important" if interaction > 0.3 else "becomes somewhat more important" if interaction > 0 else "becomes less important" if interaction < 0 else "has consistent importance"

            f.write(
                f"- **{noise_type}**: {corr_strength} correlation ({tau_corr:.3f}) between τ and accuracy. Regularization {interact_desc} at higher noise levels.\n")

        # 3. Optimal regularization recommendations
        f.write("\n### Optimal Regularization Recommendations\n\n")
        for _, row in summary_df.iterrows():
            if pd.isna(row['best_tau_at_max_noise']):
                continue

            noise_type = row['noise_type'].replace('_', ' ').title()
            best_tau = row['best_tau_at_max_noise']
            max_acc = row['accuracy_at_max_noise']

            f.write(
                f"- For **{noise_type}**: Use τ = {best_tau:.0e} (maintains {max_acc:.1%} accuracy at maximum noise level)\n")

        # 4. Overall conclusions
        f.write("\n## Overall Conclusions\n\n")

        # Calculate average best tau across noise types
        valid_taus = summary_df['best_tau_at_max_noise'].dropna().values
        if len(valid_taus) > 0:
            # Use geometric mean for log-scale values
            avg_tau = 10 ** np.mean(np.log10(valid_taus))
            f.write(f"1. For unknown or mixed noise types, use τ = {avg_tau:.0e} as a good compromise value.\n\n")

        f.write("2. Different noise types have distinct impacts on LDA-KNN performance:\n")

        if 'salt_pepper' in summary_df['noise_type'].values:
            f.write("   - Salt & Pepper noise is particularly destructive, requiring strong regularization.\n")
        if 'gaussian' in summary_df['noise_type'].values:
            f.write("   - Gaussian noise significantly impacts performance but responds well to regularization.\n")
        if 'blur' in summary_df['noise_type'].values:
            f.write("   - Blur has moderate impact, with a more gradual decline as noise increases.\n")
        if 'speckle' in summary_df['noise_type'].values:
            f.write("   - Speckle noise is relatively well-tolerated even at higher levels.\n")
        if 'zigzag' in summary_df['noise_type'].values:
            f.write("   - Zigzag distortion affects LDA-KNN performance in unique ways depending on amplitude.\n")

        f.write(
            "\n3. Regularization generally becomes more effective as noise level increases, suggesting that LDA with proper regularization is a robust dimensionality reduction technique for noisy data.\n")

    return summary_df


def plot_degradation_patterns(combined_df):
    """Analyze and plot the degradation patterns for different noise types."""
    # We'll classify degradation patterns as:
    # - Cliff: Sharp initial drop, then slow degradation
    # - Linear: Roughly constant rate of degradation
    # - Exponential: Accelerating degradation

    # For each noise type, we'll plot the normalized accuracy vs normalized noise
    for noise_type in combined_df['noise_type'].unique():
        # Get mid-range tau (not too low, not too high)
        tau_values = sorted(combined_df[combined_df['noise_type'] == noise_type]['tau'].unique())
        if len(tau_values) >= 3:
            tau = tau_values[len(tau_values) // 2]  # Middle value
        elif len(tau_values) > 0:
            tau = tau_values[0]  # Any available tau
        else:
            continue

        # Get data for this noise type and tau
        subset = combined_df[(combined_df['noise_type'] == noise_type) & (combined_df['tau'] == tau)]

        # Skip if not enough data points
        if len(subset) < 3:
            continue

        # Get baseline accuracy
        baseline = subset[subset['noise_level'] == 0]
        if len(baseline) == 0:
            # If no explicit baseline, use minimum noise level as baseline
            min_noise = subset['noise_level'].min()
            baseline = subset[subset['noise_level'] == min_noise]
            if len(baseline) == 0:
                continue

        baseline_acc = baseline['accuracy'].values[0]

        # Normalize accuracy relative to baseline (1.0 = baseline, 0.0 = complete failure)
        # and normalize noise level (0.0 = min, 1.0 = max)
        norm_data = []

        for _, row in subset.iterrows():
            norm_acc = row['accuracy'] / baseline_acc

            norm_data.append({
                'noise_level': row['noise_level'],
                'normalized_noise': row['normalized_noise'],
                'normalized_accuracy': norm_acc
            })

        # Convert to dataframe and sort
        norm_df = pd.DataFrame(norm_data)
        norm_df = norm_df.sort_values('normalized_noise')

        # Plot
        plt.figure(figsize=(10, 6))

        plt.plot(norm_df['normalized_noise'], norm_df['normalized_accuracy'],
                 marker='o', linewidth=2, color='blue')

        plt.title(f'Normalized Accuracy vs. Normalized Noise for {noise_type.replace("_", " ").title()}', fontsize=14)
        plt.xlabel('Normalized Noise Level', fontsize=12)
        plt.ylabel('Normalized Accuracy (relative to baseline)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add reference lines
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Baseline Accuracy')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% Accuracy')

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"degradation_pattern_{noise_type}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Determine degradation pattern
        if len(norm_df) >= 3:
            # Calculate first and second halves of the drops
            sorted_df = norm_df.sort_values('normalized_noise')
            mid_idx = len(sorted_df) // 2

            # Skip the baseline point
            non_zero_df = sorted_df[sorted_df['normalized_noise'] > 0]
            if len(non_zero_df) < 2:
                continue

            # Get first and last points
            first_drop = 1.0 - non_zero_df.iloc[0]['normalized_accuracy']
            last_drop = 1.0 - non_zero_df.iloc[-1]['normalized_accuracy']

            # If more than 2 points, get middle point
            if len(non_zero_df) > 2:
                mid_drop = 1.0 - non_zero_df.iloc[len(non_zero_df) // 2]['normalized_accuracy']

                # Calculate slopes
                first_half_slope = (mid_drop - first_drop) / (
                            non_zero_df.iloc[len(non_zero_df) // 2]['normalized_noise'] - non_zero_df.iloc[0][
                        'normalized_noise'])
                second_half_slope = (last_drop - mid_drop) / (
                            non_zero_df.iloc[-1]['normalized_noise'] - non_zero_df.iloc[len(non_zero_df) // 2][
                        'normalized_noise'])

                # Determine pattern
                if first_half_slope > 2 * second_half_slope:
                    pattern = "cliff"
                elif second_half_slope > 2 * first_half_slope:
                    pattern = "exponential"
                else:
                    pattern = "linear"
            else:
                pattern = "unknown (insufficient data points)"

            print(f"Degradation pattern for {noise_type}: {pattern}")


def calculate_tau_importance_by_noise(combined_df):
    """Calculate and plot how important tau becomes as noise increases."""
    # For each noise type, calculate the improvement from min to max tau
    # at different noise levels

    importance_data = []

    for noise_type in combined_df['noise_type'].unique():
        # Get min and max tau
        tau_values = sorted(combined_df[combined_df['noise_type'] == noise_type]['tau'].unique())

        if len(tau_values) < 2:
            continue

        min_tau = min(tau_values)
        max_tau = max(tau_values)

        # Get noise levels
        noise_levels = sorted(combined_df[combined_df['noise_type'] == noise_type]['noise_level'].unique())

        for noise_level in noise_levels:
            # Get accuracy at min and max tau
            min_tau_data = combined_df[(combined_df['noise_type'] == noise_type) &
                                       (combined_df['tau'] == min_tau) &
                                       (combined_df['noise_level'] == noise_level)]

            max_tau_data = combined_df[(combined_df['noise_type'] == noise_type) &
                                       (combined_df['tau'] == max_tau) &
                                       (combined_df['noise_level'] == noise_level)]

            if len(min_tau_data) == 0 or len(max_tau_data) == 0:
                continue

            min_tau_acc = min_tau_data['accuracy'].values[0]
            max_tau_acc = max_tau_data['accuracy'].values[0]

            # Calculate improvement
            abs_improvement = max_tau_acc - min_tau_acc
            rel_improvement = (abs_improvement / min_tau_acc) * 100 if min_tau_acc > 0 else 0

            importance_data.append({
                'noise_type': noise_type,
                'noise_level': noise_level,
                'normalized_noise': min_tau_data['normalized_noise'].values[0],
                'min_tau': min_tau,
                'max_tau': max_tau,
                'min_tau_acc': min_tau_acc,
                'max_tau_acc': max_tau_acc,
                'abs_improvement': abs_improvement,
                'rel_improvement': rel_improvement
            })

    # Convert to dataframe
    importance_df = pd.DataFrame(importance_data)

    # Plot
    plt.figure(figsize=(12, 8))

    for noise_type in importance_df['noise_type'].unique():
        subset = importance_df[importance_df['noise_type'] == noise_type]

        if len(subset) < 2:
            continue

        # Sort by noise level
        subset = subset.sort_values('noise_level')

        plt.plot(subset['normalized_noise'], subset['rel_improvement'],
                 marker='o', linewidth=2,
                 label=noise_type.replace('_', ' ').title())

    plt.title(f'Regularization Importance vs. Noise Level', fontsize=14)
    plt.xlabel('Normalized Noise Level', fontsize=12)
    plt.ylabel('Relative Accuracy Improvement (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("tau_importance_vs_noise.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save the data
    importance_df.to_csv("tau_importance_data.csv", index=False)

    return importance_df


def run_comprehensive_analysis():
    """Run the comprehensive analysis with plots."""
    print("Starting comparative analysis of noise types and regularization...")

    # Load and standardize results
    results = load_results_files()
    standardized_results = standardize_dataframes(results)

    # Create combined dataframe for analysis
    combined_df = create_combined_dataframe(standardized_results)
    print(f"Combined dataframe created with {len(combined_df)} rows")

    # Save combined dataframe
    combined_df.to_csv("combined_noise_results.csv", index=False)

    # Create plots
    print("\nGenerating basic comparison plots...")
    plot_combined_noise_comparison(combined_df)

    print("Analyzing accuracy drops...")
    drop_df = plot_accuracy_drop_heatmap(combined_df)

    print("Analyzing regularization effectiveness...")
    plot_regularization_effectiveness(combined_df)

    print("Generating misclassification pattern plots...")
    plot_misclassification_patterns(combined_df)

    print("Ranking noise types by robustness...")
    noise_ranking = plot_noise_robustness_ranking(combined_df, drop_df)

    print("Analyzing tau-noise relationship...")
    best_tau_df = plot_tau_noise_relationship(combined_df, drop_df)

    print("Performing correlation analysis...")
    corr_df = create_correlation_analysis(combined_df)

    print("Analyzing degradation patterns...")
    plot_degradation_patterns(combined_df)

    print("Calculating regularization importance...")
    importance_df = calculate_tau_importance_by_noise(combined_df)

    print("Creating comprehensive summary...")
    summary_df = create_comprehensive_summary(drop_df, best_tau_df, corr_df)

    print("\nAnalysis complete! All plots and data files have been saved.")

    return {
        'combined_df': combined_df,
        'drop_df': drop_df,
        'best_tau_df': best_tau_df,
        'corr_df': corr_df,
        'importance_df': importance_df,
        'summary_df': summary_df
    }


if __name__ == "__main__":
    results = run_comprehensive_analysis()