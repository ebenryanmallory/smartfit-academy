// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson: LessonData = {
  id: "c-data-science-fundamentals",
  title: "Advanced Data Science: Statistical Learning, Computational Methods, and Research Applications",
  description: "Comprehensive exploration of advanced data science methodologies, theoretical foundations, scalable algorithms, and cutting-edge research applications for graduate-level study and professional research.",
  sections: [
    {
      title: "Mathematical Foundations of Data Science",
      content: `Data science at the graduate level requires deep mathematical foundations spanning multiple disciplines:

**Statistical Learning Theory**
- **Empirical Risk Minimization**: L(f) = (1/n)∑ℓ(f(xᵢ), yᵢ) + Ω(f)
- **Rademacher Complexity**: Measures capacity of function class for generalization bounds
- **Concentration Inequalities**: McDiarmid's, Hoeffding's for finite-sample analysis
- **Uniform Convergence**: sup|L̂(f) - L(f)| → 0 as n → ∞ with high probability

**Information Theory and Entropy**
- **Differential Entropy**: h(X) = -∫ p(x) log p(x) dx for continuous distributions
- **Mutual Information**: I(X;Y) = ∫∫ p(x,y) log(p(x,y)/(p(x)p(y))) dx dy
- **KL Divergence**: D_KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
- **Maximum Entropy Principle**: Choose distribution maximizing entropy subject to constraints

**Computational Complexity and Scalability**
- **Big O Analysis**: Time/space complexity for large-scale data processing
- **Distributed Computing**: MapReduce paradigm, parallel algorithms
- **Streaming Algorithms**: Processing data with limited memory (Count-Min Sketch, HyperLogLog)
- **Approximation Algorithms**: Trading accuracy for computational efficiency

The theoretical foundation ensures principled approaches to data analysis with provable guarantees rather than heuristic methods.`
    },
    {
      title: "Advanced Statistical Inference and Uncertainty Quantification",
      content: `Modern data science requires sophisticated statistical methods for robust inference and uncertainty assessment:

\`\`\`python interactive
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import pandas as pd
from typing import Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class InferenceResult:
    """Container for statistical inference results."""
    estimate: np.ndarray
    confidence_interval: Tuple[np.ndarray, np.ndarray]
    p_value: float
    test_statistic: float
    method: str

class AdvancedStatisticalInference:
    """Advanced statistical inference methods for data science."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Significance level
        
    def bootstrap_inference(self, data: np.ndarray, statistic: Callable,
                          n_bootstrap: int = 10000, 
                          confidence_level: float = 0.95) -> InferenceResult:
        """Bootstrap inference with bias correction and acceleration (BCa)."""
        n = len(data)
        original_stat = statistic(data)
        
        # Generate bootstrap samples
        bootstrap_stats = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats[i] = statistic(bootstrap_sample)
        
        # Bias correction
        bias_correction = stats.norm.ppf((bootstrap_stats < original_stat).mean())
        
        # Acceleration correction (jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jackknife_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jackknife_sample)
        
        jackknife_mean = np.mean(jackknife_stats)
        acceleration = np.sum((jackknife_mean - jackknife_stats)**3) / \\
                      (6 * (np.sum((jackknife_mean - jackknife_stats)**2))**1.5)
        
        # BCa confidence interval
        alpha_low = (1 - confidence_level) / 2
        alpha_high = (1 + confidence_level) / 2
        
        z_low = stats.norm.ppf(alpha_low)
        z_high = stats.norm.ppf(alpha_high)
        
        adjusted_low = stats.norm.cdf(bias_correction + (bias_correction + z_low) / 
                                     (1 - acceleration * (bias_correction + z_low)))
        adjusted_high = stats.norm.cdf(bias_correction + (bias_correction + z_high) / 
                                      (1 - acceleration * (bias_correction + z_high)))
        
        ci_low = np.percentile(bootstrap_stats, 100 * adjusted_low)
        ci_high = np.percentile(bootstrap_stats, 100 * adjusted_high)
        
        # Bootstrap hypothesis test (if testing against specific value)
        p_value = 2 * min((bootstrap_stats >= original_stat).mean(),
                         (bootstrap_stats <= original_stat).mean())
        
        return InferenceResult(
            estimate=original_stat,
            confidence_interval=(ci_low, ci_high),
            p_value=p_value,
            test_statistic=original_stat,
            method="Bootstrap BCa"
        )
    
    def permutation_test(self, group1: np.ndarray, group2: np.ndarray,
                        statistic: Callable, n_permutations: int = 10000) -> InferenceResult:
        """Exact permutation test for two-sample comparison."""
        combined = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)
        observed_stat = statistic(group1, group2)
        
        # Generate permutation distribution
        permutation_stats = np.zeros(n_permutations)
        for i in range(n_permutations):
            permuted = np.random.permutation(combined)
            perm_group1 = permuted[:n1]
            perm_group2 = permuted[n1:]
            permutation_stats[i] = statistic(perm_group1, perm_group2)
        
        # Compute p-value
        p_value = (np.abs(permutation_stats) >= np.abs(observed_stat)).mean()
        
        # Approximate confidence interval using permutation distribution
        ci_low = np.percentile(permutation_stats, 2.5)
        ci_high = np.percentile(permutation_stats, 97.5)
        
        return InferenceResult(
            estimate=observed_stat,
            confidence_interval=(ci_low, ci_high),
            p_value=p_value,
            test_statistic=observed_stat,
            method="Permutation Test"
        )
    
    def bayesian_inference(self, data: np.ndarray, prior_params: Dict[str, float],
                          likelihood: str = "normal") -> Dict[str, Any]:
        """Bayesian inference with conjugate priors."""
        if likelihood == "normal":
            # Normal likelihood with normal-gamma prior
            mu_0, lambda_0 = prior_params["mu_0"], prior_params["lambda_0"]
            alpha_0, beta_0 = prior_params["alpha_0"], prior_params["beta_0"]
            
            n = len(data)
            x_bar = np.mean(data)
            s_squared = np.var(data, ddof=1)
            
            # Posterior parameters
            lambda_n = lambda_0 + n
            mu_n = (lambda_0 * mu_0 + n * x_bar) / lambda_n
            alpha_n = alpha_0 + n / 2
            beta_n = beta_0 + 0.5 * ((n - 1) * s_squared + 
                                   (lambda_0 * n * (x_bar - mu_0)**2) / lambda_n)
            
            # Posterior predictive distribution
            # t-distribution with 2*alpha_n degrees of freedom
            dof = 2 * alpha_n
            scale = np.sqrt(beta_n * (lambda_n + 1) / (alpha_n * lambda_n))
            
            return {
                "posterior_mean": mu_n,
                "posterior_precision": lambda_n,
                "posterior_shape": alpha_n,
                "posterior_rate": beta_n,
                "predictive_distribution": stats.t(df=dof, loc=mu_n, scale=scale),
                "credible_interval": stats.t.interval(0.95, df=dof, loc=mu_n, scale=scale)
            }
        
        else:
            raise ValueError(f"Likelihood '{likelihood}' not implemented")

class RobustStatistics:
    """Robust statistical methods for contaminated data."""
    
    @staticmethod
    def huber_loss(residuals: np.ndarray, delta: float = 1.35) -> np.ndarray:
        """Huber loss function for robust regression."""
        abs_residuals = np.abs(residuals)
        return np.where(abs_residuals <= delta,
                       0.5 * residuals**2,
                       delta * abs_residuals - 0.5 * delta**2)
    
    @staticmethod
    def mad_scale(data: np.ndarray) -> float:
        """Median Absolute Deviation scale estimator."""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return 1.4826 * mad  # Consistency factor for normal distribution
    
    def robust_regression(self, X: np.ndarray, y: np.ndarray, 
                         loss: str = "huber") -> Dict[str, Any]:
        """Robust regression using iteratively reweighted least squares."""
        def huber_weights(residuals: np.ndarray, delta: float = 1.35) -> np.ndarray:
            abs_residuals = np.abs(residuals)
            return np.where(abs_residuals <= delta, 1.0, delta / abs_residuals)
        
        # Initialize with ordinary least squares
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        for iteration in range(20):  # IRLS iterations
            residuals = y - X @ beta
            scale = self.mad_scale(residuals)
            
            if loss == "huber":
                weights = huber_weights(residuals / scale)
            else:
                raise ValueError(f"Loss function '{loss}' not implemented")
            
            # Weighted least squares
            W = np.diag(weights)
            beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
            
            # Check convergence
            if np.linalg.norm(beta_new - beta) < 1e-6:
                break
                
            beta = beta_new
        
        # Compute robust standard errors
        residuals = y - X @ beta
        scale = self.mad_scale(residuals)
        weights = huber_weights(residuals / scale)
        
        # Sandwich estimator for robust standard errors
        W = np.diag(weights)
        meat = X.T @ np.diag(residuals**2) @ X
        bread = np.linalg.inv(X.T @ W @ X)
        robust_cov = bread @ meat @ bread
        
        return {
            "coefficients": beta,
            "robust_standard_errors": np.sqrt(np.diag(robust_cov)),
            "residuals": residuals,
            "scale": scale,
            "weights": weights,
            "iterations": iteration + 1
        }

# Demonstration of advanced statistical inference
print("Advanced Statistical Inference Demonstration:")
print("="*50)

# Generate example data with outliers
np.random.seed(42)
n_clean = 100
n_outliers = 10

# Clean data from normal distribution
clean_data = np.random.normal(10, 2, n_clean)
# Outliers from different distribution
outliers = np.random.normal(20, 1, n_outliers)
contaminated_data = np.concatenate([clean_data, outliers])

print(f"Dataset: {len(contaminated_data)} observations")
print(f"True mean (clean): {np.mean(clean_data):.3f}")
print(f"Contaminated mean: {np.mean(contaminated_data):.3f}")

# Statistical inference
inference = AdvancedStatisticalInference()

# Bootstrap inference for median (robust estimator)
def median_statistic(data):
    return np.median(data)

bootstrap_result = inference.bootstrap_inference(contaminated_data, median_statistic)
print(f"\\nBootstrap Median Inference:")
print(f"Estimate: {bootstrap_result.estimate:.3f}")
print(f"95% CI: [{bootstrap_result.confidence_interval[0]:.3f}, {bootstrap_result.confidence_interval[1]:.3f}]")

# Bayesian inference for normal model
prior_params = {"mu_0": 10, "lambda_0": 1, "alpha_0": 1, "beta_0": 1}
bayesian_result = inference.bayesian_inference(clean_data, prior_params)
print(f"\\nBayesian Inference (Clean Data):")
print(f"Posterior mean: {bayesian_result['posterior_mean']:.3f}")
print(f"95% Credible interval: [{bayesian_result['credible_interval'][0]:.3f}, {bayesian_result['credible_interval'][1]:.3f}]")

# Robust regression example
X = np.column_stack([np.ones(len(contaminated_data)), 
                    np.random.normal(0, 1, len(contaminated_data))])
true_beta = np.array([10, 2])
y = X @ true_beta + np.random.normal(0, 1, len(contaminated_data))
# Add outliers to y
y[-n_outliers:] += np.random.normal(10, 2, n_outliers)

robust_stats = RobustStatistics()
robust_result = robust_stats.robust_regression(X, y)

# Compare with ordinary least squares
ols_beta = np.linalg.lstsq(X, y, rcond=None)[0]

print(f"\\nRobust vs OLS Regression:")
print(f"True coefficients: {true_beta}")
print(f"OLS estimates: {ols_beta}")
print(f"Robust estimates: {robust_result['coefficients']}")
print(f"Robust standard errors: {robust_result['robust_standard_errors']}")
\`\`\``
    },
    {
      title: "High-Dimensional Data Analysis and Regularization",
      content: `High-dimensional data presents unique challenges requiring specialized techniques for feature selection, regularization, and dimensionality reduction:

\`\`\`python interactive
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.decomposition import PCA, SparsePCA, FastICA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class HighDimensionalAnalysis:
    """Advanced methods for high-dimensional data analysis."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_sparse_data(self, n_samples: int = 200, n_features: int = 1000,
                           n_informative: int = 50, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate high-dimensional sparse regression data."""
        # True sparse coefficient vector
        true_coef = np.zeros(n_features)
        informative_idx = np.random.choice(n_features, n_informative, replace=False)
        true_coef[informative_idx] = np.random.normal(0, 1, n_informative)
        
        # Feature matrix with correlation structure
        # Create blocks of correlated features
        block_size = 20
        n_blocks = n_features // block_size
        
        X = np.zeros((n_samples, n_features))
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, n_features)
            block_features = end_idx - start_idx
            
            # Generate correlated block
            base_feature = np.random.normal(0, 1, n_samples)
            for j in range(block_features):
                correlation = 0.7 ** j  # Decreasing correlation
                X[:, start_idx + j] = (correlation * base_feature + 
                                      np.sqrt(1 - correlation**2) * np.random.normal(0, 1, n_samples))
        
        # Response variable
        y = X @ true_coef + np.random.normal(0, noise, n_samples)
        
        return X, y, true_coef
    
    def regularization_path(self, X: np.ndarray, y: np.ndarray, 
                          method: str = "lasso", n_alphas: int = 100) -> Dict[str, Any]:
        """Compute regularization path for different penalty parameters."""
        if method == "lasso":
            model = Lasso(max_iter=10000)
            alphas = np.logspace(-4, 1, n_alphas)
        elif method == "ridge":
            model = Ridge()
            alphas = np.logspace(-2, 3, n_alphas)
        elif method == "elastic_net":
            model = ElasticNet(l1_ratio=0.5, max_iter=10000)
            alphas = np.logspace(-4, 1, n_alphas)
        else:
            raise ValueError(f"Method {method} not supported")
        
        coef_path = np.zeros((len(alphas), X.shape[1]))
        mse_path = np.zeros(len(alphas))
        
        for i, alpha in enumerate(alphas):
            model.alpha = alpha
            model.fit(X, y)
            coef_path[i] = model.coef_
            mse_path[i] = mean_squared_error(y, model.predict(X))
        
        return {
            "alphas": alphas,
            "coefficients": coef_path,
            "mse": mse_path,
            "method": method
        }
    
    def stability_selection(self, X: np.ndarray, y: np.ndarray,
                          alpha: float = 0.01, n_bootstrap: int = 100,
                          selection_threshold: float = 0.6) -> Dict[str, Any]:
        """Stability selection for feature selection."""
        n_features = X.shape[1]
        selection_probabilities = np.zeros(n_features)
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[bootstrap_idx]
            y_boot = y[bootstrap_idx]
            
            # Fit Lasso
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X_boot, y_boot)
            
            # Record selected features
            selected = np.abs(lasso.coef_) > 1e-6
            selection_probabilities += selected
        
        selection_probabilities /= n_bootstrap
        stable_features = selection_probabilities >= selection_threshold
        
        return {
            "selection_probabilities": selection_probabilities,
            "stable_features": stable_features,
            "n_selected": np.sum(stable_features),
            "threshold": selection_threshold
        }
    
    def manifold_learning_comparison(self, X: np.ndarray, 
                                   methods: List[str] = None) -> Dict[str, np.ndarray]:
        """Compare different manifold learning techniques."""
        if methods is None:
            methods = ["pca", "sparse_pca", "ica", "tsne", "isomap"]
        
        n_components = min(50, X.shape[1])  # Limit for computational efficiency
        results = {}
        
        if "pca" in methods:
            pca = PCA(n_components=n_components, random_state=self.random_state)
            results["pca"] = pca.fit_transform(X)
            
        if "sparse_pca" in methods:
            sparse_pca = SparsePCA(n_components=min(10, n_components), 
                                 random_state=self.random_state, max_iter=100)
            results["sparse_pca"] = sparse_pca.fit_transform(X)
            
        if "ica" in methods:
            ica = FastICA(n_components=min(20, n_components), 
                         random_state=self.random_state, max_iter=200)
            results["ica"] = ica.fit_transform(X)
            
        if "tsne" in methods and X.shape[0] <= 1000:  # t-SNE is slow for large datasets
            tsne = TSNE(n_components=2, random_state=self.random_state, 
                       perplexity=min(30, X.shape[0]//4))
            results["tsne"] = tsne.fit_transform(X)
            
        if "isomap" in methods and X.shape[0] <= 1000:
            isomap = Isomap(n_components=min(10, n_components), 
                           n_neighbors=min(10, X.shape[0]//10))
            results["isomap"] = isomap.fit_transform(X)
        
        return results

class InformationTheoreticFeatureSelection:
    """Feature selection using information theory measures."""
    
    @staticmethod
    def mutual_information_matrix(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Compute mutual information matrix between features."""
        n_features = X.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        
        # Discretize continuous features
        X_discrete = np.zeros_like(X, dtype=int)
        for i in range(n_features):
            X_discrete[:, i] = np.digitize(X[:, i], np.linspace(X[:, i].min(), X[:, i].max(), n_bins))
        
        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:
                    mi_matrix[i, j] = 1.0  # Self-information normalized to 1
                else:
                    # Compute mutual information
                    mi = mutual_info_regression(X_discrete[:, [i]], X_discrete[:, j])[0]
                    mi_matrix[i, j] = mi_matrix[j, i] = mi
        
        return mi_matrix
    
    def maximal_information_coefficient(self, x: np.ndarray, y: np.ndarray,
                                      alpha: float = 0.6) -> float:
        """Compute Maximal Information Coefficient (MIC)."""
        # Simplified MIC implementation
        n = len(x)
        max_grid_size = int(n**alpha)
        
        max_mic = 0
        for grid_x in range(2, min(max_grid_size, 10)):  # Limit for efficiency
            for grid_y in range(2, min(max_grid_size, 10)):
                # Discretize variables
                x_bins = np.linspace(x.min(), x.max(), grid_x + 1)
                y_bins = np.linspace(y.min(), y.max(), grid_y + 1)
                
                x_disc = np.digitize(x, x_bins) - 1
                y_disc = np.digitize(y, y_bins) - 1
                
                # Compute mutual information
                mi = mutual_info_regression(x_disc.reshape(-1, 1), y_disc)[0]
                
                # Normalize by log of minimum grid dimension
                normalized_mi = mi / np.log(min(grid_x, grid_y))
                max_mic = max(max_mic, normalized_mi)
        
        return max_mic

# Demonstration
print("High-Dimensional Data Analysis Demonstration:")
print("="*50)

# Generate high-dimensional sparse data
hd_analysis = HighDimensionalAnalysis()
X, y, true_coef = hd_analysis.generate_sparse_data(n_samples=200, n_features=500, 
                                                  n_informative=25)

print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"True number of informative features: {np.sum(true_coef != 0)}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Regularization path analysis
print("\\n1. Regularization Path Analysis:")
lasso_path = hd_analysis.regularization_path(X_scaled, y, method="lasso")

# Find optimal alpha using cross-validation
lasso_cv = GridSearchCV(Lasso(max_iter=10000), 
                       {"alpha": lasso_path["alphas"]}, 
                       cv=5, scoring="neg_mean_squared_error")
lasso_cv.fit(X_scaled, y)
optimal_alpha = lasso_cv.best_params_["alpha"]

print(f"Optimal Lasso alpha: {optimal_alpha:.4f}")
print(f"Best CV score: {-lasso_cv.best_score_:.4f}")

# Final model with optimal alpha
final_lasso = Lasso(alpha=optimal_alpha, max_iter=10000)
final_lasso.fit(X_scaled, y)
selected_features = np.abs(final_lasso.coef_) > 1e-6
print(f"Number of selected features: {np.sum(selected_features)}")

# 2. Stability selection
print("\\n2. Stability Selection:")
stability_result = hd_analysis.stability_selection(X_scaled, y, alpha=optimal_alpha)
print(f"Number of stable features: {stability_result['n_selected']}")
print(f"Selection threshold: {stability_result['threshold']}")

# 3. Feature importance ranking
print("\\n3. Feature Importance Analysis:")
feature_importance = np.abs(final_lasso.coef_)
top_features = np.argsort(feature_importance)[::-1][:10]

print("Top 10 features by Lasso coefficient magnitude:")
for i, idx in enumerate(top_features):
    print(f"Feature {idx}: coef={final_lasso.coef_[idx]:.4f}, "
          f"true_coef={true_coef[idx]:.4f}")

# 4. Manifold learning comparison
print("\\n4. Dimensionality Reduction Comparison:")
# Use subset of features for computational efficiency
X_subset = X_scaled[:, :100]
manifold_results = hd_analysis.manifold_learning_comparison(X_subset, 
                                                          ["pca", "sparse_pca", "ica"])

for method, transformed in manifold_results.items():
    print(f"{method.upper()}: reduced to {transformed.shape[1]} dimensions")
    
    # Compute reconstruction quality for linear methods
    if method in ["pca", "ica"]:
        if method == "pca":
            pca = PCA(n_components=transformed.shape[1])
            pca.fit(X_subset)
            reconstructed = pca.inverse_transform(transformed)
        elif method == "ica":
            # ICA reconstruction is more complex, simplified here
            reconstructed = transformed @ np.linalg.pinv(transformed) @ X_subset
        
        reconstruction_error = np.mean((X_subset - reconstructed)**2)
        print(f"  Reconstruction MSE: {reconstruction_error:.6f}")

# 5. Information-theoretic feature selection
print("\\n5. Information-Theoretic Analysis:")
info_selector = InformationTheoreticFeatureSelection()

# Compute MIC for top features with target
top_10_features = X_scaled[:, top_features[:10]]
mic_scores = []

for i in range(top_10_features.shape[1]):
    mic = info_selector.maximal_information_coefficient(top_10_features[:, i], y)
    mic_scores.append(mic)

print("MIC scores for top 10 features:")
for i, mic in enumerate(mic_scores):
    print(f"Feature {top_features[i]}: MIC = {mic:.4f}")

print(f"\\nMean MIC score: {np.mean(mic_scores):.4f}")
print(f"Std MIC score: {np.std(mic_scores):.4f}")
\`\`\``
    },
    {
      title: "Scalable Computing and Big Data Analytics",
      content: `Modern data science requires scalable algorithms and distributed computing approaches for massive datasets:

\`\`\`python interactive
import numpy as np
import pandas as pd
from typing import Generator, Iterator, Tuple, List, Dict, Any, Optional
from collections import defaultdict
import time
import hashlib
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class StreamingStatistics:
    """Container for streaming statistics."""
    count: int
    mean: float
    variance: float
    min_val: float
    max_val: float
    
class OnlineAlgorithms:
    """Online algorithms for streaming data analysis."""
    
    def __init__(self):
        self.statistics = {}
    
    def welford_online_variance(self, data_stream: Iterator[float], 
                               column_name: str = "default") -> StreamingStatistics:
        """Welford's online algorithm for computing mean and variance."""
        count = 0
        mean = 0.0
        M2 = 0.0  # Sum of squares of differences from mean
        min_val = float('inf')
        max_val = float('-inf')
        
        for x in data_stream:
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            M2 += delta * delta2
            min_val = min(min_val, x)
            max_val = max(max_val, x)
        
        variance = M2 / (count - 1) if count > 1 else 0.0
        
        stats = StreamingStatistics(count, mean, variance, min_val, max_val)
        self.statistics[column_name] = stats
        return stats
    
    def reservoir_sampling(self, data_stream: Iterator[Any], 
                          k: int) -> List[Any]:
        """Reservoir sampling for uniform random sampling from stream."""
        reservoir = []
        
        for i, item in enumerate(data_stream):
            if i < k:
                reservoir.append(item)
            else:
                # Random replacement with probability k/(i+1)
                j = np.random.randint(0, i + 1)
                if j < k:
                    reservoir[j] = item
        
        return reservoir

class SketchAlgorithms:
    """Probabilistic data structures for approximate computation."""
    
    def __init__(self):
        pass
    
    class CountMinSketch:
        """Count-Min Sketch for frequency estimation."""
        
        def __init__(self, width: int, depth: int):
            self.width = width
            self.depth = depth
            self.table = np.zeros((depth, width), dtype=int)
            self.hash_functions = self._generate_hash_functions()
        
        def _generate_hash_functions(self):
            """Generate pairwise independent hash functions."""
            functions = []
            for i in range(self.depth):
                a = np.random.randint(1, 2**31 - 1)
                b = np.random.randint(0, 2**31 - 1)
                functions.append((a, b))
            return functions
        
        def _hash(self, item: str, func_params: Tuple[int, int]) -> int:
            """Universal hash function."""
            a, b = func_params
            hash_val = int(hashlib.md5(item.encode()).hexdigest(), 16)
            return ((a * hash_val + b) % (2**31 - 1)) % self.width
        
        def update(self, item: str, count: int = 1):
            """Update count for item."""
            for i, func_params in enumerate(self.hash_functions):
                j = self._hash(item, func_params)
                self.table[i, j] += count
        
        def query(self, item: str) -> int:
            """Estimate count for item."""
            estimates = []
            for i, func_params in enumerate(self.hash_functions):
                j = self._hash(item, func_params)
                estimates.append(self.table[i, j])
            return min(estimates)  # Take minimum for better accuracy
    
    class HyperLogLog:
        """HyperLogLog for cardinality estimation."""
        
        def __init__(self, precision: int):
            self.precision = precision
            self.m = 2 ** precision  # Number of buckets
            self.buckets = np.zeros(self.m, dtype=int)
        
        def _hash(self, item: str) -> int:
            """Hash function for HyperLogLog."""
            return int(hashlib.md5(item.encode()).hexdigest(), 16)
        
        def add(self, item: str):
            """Add item to set."""
            hash_val = self._hash(item)
            
            # Use first 'precision' bits for bucket
            bucket = hash_val & ((1 << self.precision) - 1)
            
            # Count leading zeros in remaining bits
            remaining = hash_val >> self.precision
            leading_zeros = 1
            while remaining & 1 == 0 and remaining > 0:
                leading_zeros += 1
                remaining >>= 1
            
            # Update bucket with maximum leading zeros seen
            self.buckets[bucket] = max(self.buckets[bucket], leading_zeros)
        
        def estimate_cardinality(self) -> float:
            """Estimate number of distinct elements."""
            # Raw estimate
            raw_estimate = (0.7213 / (1 + 1.079 / self.m)) * self.m**2 / np.sum(2**(-bucket) for bucket in self.buckets)
            
            # Apply small range and large range corrections
            if raw_estimate <= 2.5 * self.m:
                # Small range correction
                zeros = np.sum(self.buckets == 0)
                if zeros != 0:
                    return self.m * np.log(self.m / zeros)
            
            if raw_estimate <= (1.0/30.0) * 2**32:
                return raw_estimate
            else:
                # Large range correction
                return -2**32 * np.log(1 - raw_estimate / 2**32)
            
            return raw_estimate

class DistributedComputing:
    """Simplified distributed computing patterns."""
    
    def __init__(self):
        pass
    
    def map_reduce_word_count(self, documents: List[str]) -> Dict[str, int]:
        """MapReduce word count implementation."""
        # Map phase
        def mapper(document: str) -> List[Tuple[str, int]]:
            words = document.lower().split()
            return [(word.strip('.,!?;'), 1) for word in words if word.strip('.,!?;')]
        
        # Shuffle phase (group by key)
        def shuffle(mapped_results: List[List[Tuple[str, int]]]) -> Dict[str, List[int]]:
            shuffled = defaultdict(list)
            for result_list in mapped_results:
                for word, count in result_list:
                    shuffled[word].append(count)
            return shuffled
        
        # Reduce phase
        def reducer(word: str, counts: List[int]) -> Tuple[str, int]:
            return word, sum(counts)
        
        # Execute MapReduce
        mapped = [mapper(doc) for doc in documents]
        shuffled = shuffle(mapped)
        reduced = {word: reducer(word, counts)[1] for word, counts in shuffled.items()}
        
        return reduced
    
    def parallel_gradient_descent(self, X_chunks: List[np.ndarray], 
                                y_chunks: List[np.ndarray],
                                learning_rate: float = 0.01,
                                n_iterations: int = 100) -> np.ndarray:
        """Simplified parallel gradient descent."""
        n_features = X_chunks[0].shape[1]
        weights = np.zeros(n_features)
        
        for iteration in range(n_iterations):
            # Compute gradients on each chunk (parallel)
            gradients = []
            for X_chunk, y_chunk in zip(X_chunks, y_chunks):
                predictions = X_chunk @ weights
                gradient = X_chunk.T @ (predictions - y_chunk) / len(X_chunk)
                gradients.append(gradient)
            
            # Aggregate gradients (reduce)
            average_gradient = np.mean(gradients, axis=0)
            
            # Update weights
            weights -= learning_rate * average_gradient
        
        return weights

class MemoryEfficientProcessing:
    """Memory-efficient data processing techniques."""
    
    def __init__(self):
        pass
    
    def chunked_processing(self, data_generator: Generator[np.ndarray, None, None],
                          processing_function: callable,
                          chunk_size: int = 1000) -> Generator[Any, None, None]:
        """Process data in chunks to manage memory usage."""
        chunk = []
        
        for item in data_generator:
            chunk.append(item)
            
            if len(chunk) >= chunk_size:
                # Process chunk
                result = processing_function(np.array(chunk))
                yield result
                chunk = []
        
        # Process remaining items
        if chunk:
            result = processing_function(np.array(chunk))
            yield result
    
    def memory_mapped_processing(self, large_array: np.ndarray,
                                processing_function: callable,
                                chunk_size: int = 10000) -> Generator[Any, None, None]:
        """Process large arrays using memory mapping."""
        n_samples = large_array.shape[0]
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = large_array[start_idx:end_idx]
            
            result = processing_function(chunk)
            yield result

# Demonstration
print("Scalable Computing and Big Data Analytics Demonstration:")
print("="*60)

# 1. Streaming algorithms
print("1. Streaming Algorithms:")
online_alg = OnlineAlgorithms()

# Generate streaming data
def generate_streaming_data(n_samples: int = 10000) -> Generator[float, None, None]:
    """Generate streaming normal data with concept drift."""
    for i in range(n_samples):
        # Concept drift: mean changes over time
        mean = 5 + 3 * np.sin(2 * np.pi * i / 1000)
        yield np.random.normal(mean, 2)

# Compute streaming statistics
streaming_data = generate_streaming_data(10000)
stats = online_alg.welford_online_variance(streaming_data)
print(f"Streaming statistics:")
print(f"  Count: {stats.count}")
print(f"  Mean: {stats.mean:.4f}")
print(f"  Variance: {stats.variance:.4f}")
print(f"  Range: [{stats.min_val:.4f}, {stats.max_val:.4f}]")

# 2. Probabilistic data structures
print("\\n2. Probabilistic Data Structures:")
sketch_alg = SketchAlgorithms()

# Count-Min Sketch for frequency estimation
cms = sketch_alg.CountMinSketch(width=1000, depth=5)
words = ["apple", "banana", "apple", "cherry", "apple", "banana", "date"] * 1000
for word in words:
    cms.update(word)

print("Count-Min Sketch frequency estimates:")
unique_words = ["apple", "banana", "cherry", "date", "elderberry"]
for word in unique_words:
    estimated_count = cms.query(word)
    true_count = words.count(word)
    print(f"  {word}: estimated={estimated_count}, true={true_count}")

# HyperLogLog for cardinality estimation
hll = sketch_alg.HyperLogLog(precision=10)
items = [f"item_{i}" for i in range(10000)]
# Add some duplicates
items.extend([f"item_{i}" for i in range(5000)])

for item in items:
    hll.add(item)

estimated_cardinality = hll.estimate_cardinality()
true_cardinality = len(set(items))
print(f"\\nHyperLogLog cardinality estimation:")
print(f"  Estimated: {estimated_cardinality:.0f}")
print(f"  True: {true_cardinality}")
print(f"  Error: {abs(estimated_cardinality - true_cardinality):.0f}")

# 3. Distributed computing patterns
print("\\n3. Distributed Computing:")
dist_comp = DistributedComputing()

# MapReduce word count
documents = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps under the warm sun",
    "Quick brown foxes are very clever animals",
    "Dogs and foxes are both mammals that jump"
]

word_counts = dist_comp.map_reduce_word_count(documents)
print("MapReduce word count (top 10):")
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
for word, count in sorted_words[:10]:
    print(f"  {word}: {count}")

# 4. Memory-efficient processing
print("\\n4. Memory-Efficient Processing:")
memory_proc = MemoryEfficientProcessing()

# Generate large dataset simulation
def large_data_generator(n_samples: int = 100000) -> Generator[np.ndarray, None, None]:
    """Simulate large dataset generation."""
    for i in range(0, n_samples, 1000):
        chunk = np.random.normal(0, 1, (1000, 50))  # 1000 samples, 50 features
        yield chunk

# Process in chunks
def compute_chunk_statistics(chunk: np.ndarray) -> Dict[str, float]:
    """Compute statistics for a chunk."""
    return {
        "mean": np.mean(chunk),
        "std": np.std(chunk),
        "samples": chunk.shape[0]
    }

# Process streaming chunks
chunk_results = list(memory_proc.chunked_processing(
    large_data_generator(10000), 
    compute_chunk_statistics,
    chunk_size=1000
))

# Aggregate results
total_samples = sum(result["samples"] for result in chunk_results)
weighted_mean = sum(result["mean"] * result["samples"] for result in chunk_results) / total_samples
print(f"Chunked processing results:")
print(f"  Total samples processed: {total_samples}")
print(f"  Weighted mean: {weighted_mean:.6f}")
print(f"  Number of chunks: {len(chunk_results)}")

# 5. Performance comparison
print("\\n5. Performance Analysis:")

# Compare streaming vs batch processing
def batch_statistics(data: np.ndarray) -> Dict[str, float]:
    """Compute statistics using batch processing."""
    return {
        "mean": np.mean(data),
        "variance": np.var(data, ddof=1),
        "min": np.min(data),
        "max": np.max(data)
    }

# Generate test data
test_data = np.random.normal(5, 2, 50000)

# Batch processing time
start_time = time.time()
batch_stats = batch_statistics(test_data)
batch_time = time.time() - start_time

# Streaming processing time
start_time = time.time()
stream_stats = online_alg.welford_online_variance(iter(test_data), "test")
stream_time = time.time() - start_time

print(f"Performance comparison (50,000 samples):")
print(f"  Batch processing time: {batch_time:.6f} seconds")
print(f"  Streaming processing time: {stream_time:.6f} seconds")
print(f"  Memory usage: Batch O(n), Streaming O(1)")
print(f"  Mean difference: {abs(batch_stats['mean'] - stream_stats.mean):.8f}")
\`\`\``
    },
    {
      title: "Advanced Visualization and Exploratory Data Analysis",
      content: `Graduate-level data visualization requires sophisticated techniques for high-dimensional data, uncertainty representation, and interactive exploration:

\`\`\`python interactive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from typing import Tuple, List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualization:
    """Advanced visualization techniques for data science."""
    
    def __init__(self, style: str = "whitegrid"):
        sns.set_style(style)
        self.figure_size = (12, 8)
    
    def uncertainty_visualization(self, x: np.ndarray, y_pred: np.ndarray, 
                                y_std: np.ndarray, y_true: Optional[np.ndarray] = None,
                                title: str = "Uncertainty Visualization") -> plt.Figure:
        """Visualize predictions with uncertainty bounds."""
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size)
        
        # Sort for better visualization
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        y_std_sorted = y_std[sort_idx]
        
        # Plot prediction with uncertainty bands
        ax.plot(x_sorted, y_pred_sorted, 'b-', label='Prediction', linewidth=2)
        ax.fill_between(x_sorted, 
                       y_pred_sorted - 1.96 * y_std_sorted,
                       y_pred_sorted + 1.96 * y_std_sorted,
                       alpha=0.3, color='blue', label='95% Confidence')
        ax.fill_between(x_sorted,
                       y_pred_sorted - y_std_sorted,
                       y_pred_sorted + y_std_sorted,
                       alpha=0.5, color='blue', label='68% Confidence')
        
        if y_true is not None:
            y_true_sorted = y_true[sort_idx]
            ax.plot(x_sorted, y_true_sorted, 'r--', label='True Function', linewidth=2)
        
        ax.set_xlabel('Input Variable')
        ax.set_ylabel('Output Variable')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def multidimensional_scaling_plot(self, X: np.ndarray, labels: Optional[np.ndarray] = None,
                                    method: str = "pca", title: str = "Dimensionality Reduction") -> plt.Figure:
        """Create publication-quality dimensionality reduction plots."""
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            raise ValueError(f"Method {method} not supported")
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_reduced = reducer.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Main scatter plot
        if labels is not None:
            scatter = axes[0].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                    c=labels, cmap='viridis', alpha=0.7, s=50)
            plt.colorbar(scatter, ax=axes[0])
        else:
            axes[0].scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7, s=50)
        
        axes[0].set_xlabel(f'{method.upper()} Component 1')
        axes[0].set_ylabel(f'{method.upper()} Component 2')
        axes[0].set_title(f'{title} - {method.upper()}')
        axes[0].grid(True, alpha=0.3)
        
        # Explained variance (for PCA) or density plot
        if method == "pca" and hasattr(reducer, 'explained_variance_ratio_'):
            # Explained variance plot
            cumvar = np.cumsum(reducer.explained_variance_ratio_)
            axes[1].bar(range(len(cumvar)), reducer.explained_variance_ratio_, 
                       alpha=0.7, color='skyblue')
            axes[1].plot(range(len(cumvar)), cumvar, 'ro-', linewidth=2)
            axes[1].set_xlabel('Principal Component')
            axes[1].set_ylabel('Explained Variance Ratio')
            axes[1].set_title('Explained Variance by Component')
            axes[1].grid(True, alpha=0.3)
        else:
            # Density plot for t-SNE
            axes[1].hist2d(X_reduced[:, 0], X_reduced[:, 1], bins=20, cmap='Blues')
            axes[1].set_xlabel(f'{method.upper()} Component 1')
            axes[1].set_ylabel(f'{method.upper()} Component 2')
            axes[1].set_title('Point Density')
        
        plt.tight_layout()
        return fig
    
    def correlation_heatmap_advanced(self, data: pd.DataFrame, 
                                   method: str = "pearson",
                                   cluster: bool = True) -> plt.Figure:
        """Advanced correlation heatmap with clustering and significance testing."""
        # Compute correlation matrix
        if method == "pearson":
            corr_matrix = data.corr(method='pearson')
        elif method == "spearman":
            corr_matrix = data.corr(method='spearman')
        elif method == "kendall":
            corr_matrix = data.corr(method='kendall')
        else:
            raise ValueError(f"Method {method} not supported")
        
        # Cluster correlation matrix if requested
        if cluster:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(corr_matrix)
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='average')
            
            # Get ordering from dendrogram
            dendro = dendrogram(linkage_matrix, no_plot=True)
            cluster_order = dendro['leaves']
            
            # Reorder correlation matrix
            corr_matrix = corr_matrix.iloc[cluster_order, cluster_order]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(f'{method.capitalize()} Correlation Matrix' + 
                    (' (Clustered)' if cluster else ''))
        
        plt.tight_layout()
        return fig
    
    def feature_importance_plot(self, feature_names: List[str], 
                              importances: np.ndarray,
                              errors: Optional[np.ndarray] = None,
                              top_k: int = 20) -> plt.Figure:
        """Publication-quality feature importance plot."""
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_k]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, max(6, top_k * 0.3)))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(indices))
        bars = ax.barh(y_pos, importances[indices], 
                      xerr=errors[indices] if errors is not None else None,
                      alpha=0.8, color='steelblue', capsize=3)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance Analysis')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances[indices])):
            ax.text(bar.get_width() + (max(importances) * 0.01), 
                   bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        return fig

class StatisticalVisualization:
    """Statistical visualization techniques."""
    
    @staticmethod
    def qq_plot_with_confidence(data: np.ndarray, 
                              distribution: str = "normal") -> plt.Figure:
        """Q-Q plot with confidence bands."""
        from scipy import stats
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Standard Q-Q plot
        if distribution == "normal":
            stats.probplot(data, dist="norm", plot=axes[0])
            axes[0].set_title("Q-Q Plot vs Normal Distribution")
        
        # Q-Q plot with confidence bands (simplified)
        sorted_data = np.sort(data)
        n = len(data)
        theoretical_quantiles = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
        
        axes[1].scatter(theoretical_quantiles, sorted_data, alpha=0.7)
        
        # Add confidence bands (approximate)
        se = np.std(data) / np.sqrt(n)
        lower_bound = sorted_data - 1.96 * se
        upper_bound = sorted_data + 1.96 * se
        
        axes[1].fill_between(theoretical_quantiles, lower_bound, upper_bound, 
                           alpha=0.3, color='red', label='95% Confidence Band')
        
        # Reference line
        slope, intercept = np.polyfit(theoretical_quantiles, sorted_data, 1)
        line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
        line_y = slope * line_x + intercept
        axes[1].plot(line_x, line_y, 'r-', linewidth=2, label='Best Fit Line')
        
        axes[1].set_xlabel('Theoretical Quantiles')
        axes[1].set_ylabel('Sample Quantiles')
        axes[1].set_title('Q-Q Plot with Confidence Bands')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def residual_analysis_plot(y_true: np.ndarray, y_pred: np.ndarray,
                             features: Optional[np.ndarray] = None) -> plt.Figure:
        """Comprehensive residual analysis visualization."""
        residuals = y_true - y_pred
        
        if features is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Fitted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Normal Q-Q plot of residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Normal Q-Q Plot of Residuals")
        
        # Scale-Location plot
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        axes[1, 0].scatter(y_pred, sqrt_abs_residuals, alpha=0.7)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Residuals|')
        axes[1, 0].set_title('Scale-Location Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, density=True)
        
        # Overlay normal distribution
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, np.mean(residuals), np.std(residuals))
        axes[1, 1].plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal PDF')
        
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Demonstration
print("Advanced Visualization and EDA Demonstration:")
print("="*50)

# Generate synthetic dataset for demonstration
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                          n_redundant=5, n_clusters_per_class=1, random_state=42)

# Add some noise and create a DataFrame
feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Generated dataset: {df.shape[0]} samples, {df.shape[1]-1} features")

# Initialize visualization classes
adv_viz = AdvancedVisualization()
stat_viz = StatisticalVisualization()

# 1. Advanced correlation analysis
print("\\n1. Advanced Correlation Analysis:")
corr_fig = adv_viz.correlation_heatmap_advanced(df.drop('target', axis=1), 
                                              method="pearson", cluster=True)
plt.savefig('/dev/null')  # Don't display in text output
plt.close()

# Compute correlation statistics
corr_matrix = df.drop('target', axis=1).corr()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = []

for i in range(len(upper_triangle.columns)):
    for j in range(len(upper_triangle.columns)):
        if not pd.isna(upper_triangle.iloc[i, j]) and abs(upper_triangle.iloc[i, j]) > 0.7:
            high_corr_pairs.append((upper_triangle.columns[i], 
                                  upper_triangle.columns[j], 
                                  upper_triangle.iloc[i, j]))

print(f"Number of high correlation pairs (|r| > 0.7): {len(high_corr_pairs)}")
if high_corr_pairs:
    print("Top 5 correlations:")
    sorted_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    for feat1, feat2, corr in sorted_pairs[:5]:
        print(f"  {feat1} - {feat2}: {corr:.3f}")

# 2. Dimensionality reduction visualization
print("\\n2. Dimensionality Reduction Analysis:")
pca_fig = adv_viz.multidimensional_scaling_plot(X, labels=y, method="pca", 
                                               title="PCA Visualization")
plt.savefig('/dev/null')
plt.close()

# PCA analysis
pca = PCA()
pca.fit(StandardScaler().fit_transform(X))
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumvar >= 0.95) + 1

print(f"Components needed for 95% variance: {n_components_95}")
print(f"First 5 components explain: {cumvar[4]:.3f} of variance")

# 3. Feature importance analysis
print("\\n3. Feature Importance Analysis:")
from sklearn.ensemble import RandomForestClassifier

# Train random forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
std_importances = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

importance_fig = adv_viz.feature_importance_plot(feature_names, importances, 
                                               std_importances, top_k=10)
plt.savefig('/dev/null')
plt.close()

print("Top 5 most important features:")
top_indices = np.argsort(importances)[::-1][:5]
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f} ± {std_importances[idx]:.4f}")

# 4. Statistical diagnostic plots
print("\\n4. Statistical Diagnostic Analysis:")

# Create regression example for residual analysis
from sklearn.linear_model import LinearRegression
reg_X = df[['Feature_1', 'Feature_2', 'Feature_3']].values
reg_y = 2 * reg_X[:, 0] + 1.5 * reg_X[:, 1] - 0.5 * reg_X[:, 2] + np.random.normal(0, 0.5, len(reg_X))

lr = LinearRegression()
lr.fit(reg_X, reg_y)
y_pred = lr.predict(reg_X)

# Residual analysis
residual_fig = stat_viz.residual_analysis_plot(reg_y, y_pred)
plt.savefig('/dev/null')
plt.close()

# Compute diagnostic statistics
residuals = reg_y - y_pred
rmse = np.sqrt(np.mean(residuals**2))
mae = np.mean(np.abs(residuals))
r2 = 1 - np.sum(residuals**2) / np.sum((reg_y - np.mean(reg_y))**2)

print(f"Regression diagnostics:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R²: {r2:.4f}")

# Test for normality of residuals
from scipy.stats import shapiro, jarque_bera

shapiro_stat, shapiro_p = shapiro(residuals)
jb_stat, jb_p = jarque_bera(residuals)

print(f"  Shapiro-Wilk test p-value: {shapiro_p:.4f}")
print(f"  Jarque-Bera test p-value: {jb_p:.4f}")

# 5. Uncertainty visualization example
print("\\n5. Uncertainty Quantification Visualization:")

# Create synthetic data with uncertainty
x_test = np.linspace(0, 10, 100)
y_true_func = lambda x: 2 * np.sin(x) + 0.5 * x
y_test_true = y_true_func(x_test)

# Simulate prediction with uncertainty
y_test_pred = y_test_true + np.random.normal(0, 0.1, len(x_test))
y_test_std = 0.2 + 0.1 * np.abs(x_test - 5)  # Heteroscedastic uncertainty

uncertainty_fig = adv_viz.uncertainty_visualization(x_test, y_test_pred, y_test_std, 
                                                  y_test_true, "Prediction with Uncertainty")
plt.savefig('/dev/null')
plt.close()

# Compute uncertainty metrics
coverage_68 = np.mean(np.abs(y_test_true - y_test_pred) <= y_test_std)
coverage_95 = np.mean(np.abs(y_test_true - y_test_pred) <= 1.96 * y_test_std)

print(f"Uncertainty calibration:")
print(f"  68% confidence coverage: {coverage_68:.3f} (expected: 0.68)")
print(f"  95% confidence coverage: {coverage_95:.3f} (expected: 0.95)")

mean_uncertainty = np.mean(y_test_std)
print(f"  Mean uncertainty: {mean_uncertainty:.4f}")
\`\`\``
    },
    {
      title: "Research Applications and Case Studies",
      content: `Advanced data science research spans multiple domains with specialized methodologies and evaluation frameworks:

**🧬 Computational Biology and Bioinformatics**
- **Genomic Data Analysis**: GWAS studies, sequence alignment, phylogenetic reconstruction
- **Single-Cell RNA Sequencing**: Dimensionality reduction, cell type classification, trajectory inference
- **Protein Structure Prediction**: AlphaFold-style deep learning approaches
- **Drug Discovery**: Molecular property prediction, virtual screening, QSAR modeling

**🌍 Climate Science and Environmental Modeling**
- **Weather Prediction**: Ensemble forecasting, data assimilation, numerical weather models
- **Climate Change Analysis**: Time series analysis, extreme event detection, attribution studies
- **Satellite Data Processing**: Image classification, change detection, multi-spectral analysis
- **Ecosystem Modeling**: Species distribution models, biodiversity prediction, conservation planning

**💰 Financial Engineering and Risk Management**
- **Algorithmic Trading**: High-frequency data analysis, market microstructure, regime detection
- **Risk Modeling**: Value-at-Risk, stress testing, portfolio optimization under uncertainty
- **Credit Scoring**: Survival analysis, machine learning for default prediction
- **Fraud Detection**: Anomaly detection, graph analytics, real-time decision systems

**🏥 Healthcare and Medical Informatics**
- **Medical Imaging**: CNN architectures for radiology, pathology slide analysis
- **Electronic Health Records**: Natural language processing, clinical prediction models
- **Precision Medicine**: Biomarker discovery, treatment response prediction, patient stratification
- **Epidemiological Modeling**: Disease spread simulation, intervention effectiveness analysis

**🚗 Autonomous Systems and Robotics**
- **Computer Vision**: Object detection, semantic segmentation, 3D scene understanding
- **Sensor Fusion**: LiDAR, camera, radar data integration for perception
- **Path Planning**: Reinforcement learning for navigation, SLAM algorithms
- **Human-Robot Interaction**: Intent recognition, natural language understanding

**🏭 Industrial Applications and IoT**
- **Predictive Maintenance**: Time series forecasting, survival analysis, condition monitoring
- **Quality Control**: Statistical process control, defect detection, automated inspection
- **Supply Chain Optimization**: Demand forecasting, inventory management, logistics optimization
- **Smart Manufacturing**: Digital twins, process optimization, fault diagnosis

**📚 Graduate Research Methodology**

**Literature Review and Problem Formulation**
1. **Systematic Literature Review**: Use databases (arXiv, Google Scholar, Web of Science)
2. **Gap Analysis**: Identify limitations in existing approaches
3. **Problem Formulation**: Define research questions with measurable objectives
4. **Baseline Establishment**: Implement state-of-the-art methods for comparison

**Experimental Design and Validation**
1. **Cross-Validation Strategies**: Time series CV, group CV, nested CV for model selection
2. **Statistical Testing**: Multiple comparison correction, effect size analysis
3. **Reproducibility**: Version control, environment management, random seed control
4. **Benchmark Datasets**: Use standard datasets for fair comparison

**Publication and Dissemination**
1. **Conference Papers**: NeurIPS, ICML, ICLR, KDD, AAAI for methodological contributions
2. **Journal Articles**: JMLR, Nature Machine Intelligence, IEEE TPAMI for comprehensive studies
3. **Code Release**: GitHub repositories with documentation and tutorials
4. **Open Science**: Data sharing, reproducible research practices

**🔬 Current Research Frontiers**

**Foundation Models and Large-Scale Learning**
- **Scaling Laws**: Understanding compute-performance relationships
- **Transfer Learning**: Pre-trained models for domain adaptation
- **Few-Shot Learning**: Meta-learning approaches for limited data scenarios
- **Multimodal Learning**: Vision-language models, cross-modal reasoning

**Causal Inference and Scientific Discovery**
- **Causal Discovery**: Learning causal graphs from observational data
- **Do-Calculus**: Pearl's framework for causal reasoning
- **Instrumental Variables**: Natural experiments and quasi-experimental design
- **Scientific Machine Learning**: Physics-informed neural networks, symbolic regression

**Trustworthy and Explainable AI**
- **Interpretability**: LIME, SHAP, integrated gradients for model explanation
- **Fairness**: Demographic parity, equalized odds, individual fairness metrics
- **Robustness**: Adversarial training, certified defenses, distributional robustness
- **Privacy**: Differential privacy, federated learning, secure computation

**🎯 Career Development for Graduate Students**

**Technical Skills Development**
- **Programming**: Python/R proficiency, software engineering best practices
- **Mathematics**: Linear algebra, probability, optimization, statistics
- **Computing**: Distributed systems, cloud computing, high-performance computing
- **Communication**: Technical writing, presentation skills, visualization

**Research Experience Building**
- **Internships**: Industry research labs, national laboratories
- **Collaborations**: Cross-disciplinary projects, international partnerships
- **Competitions**: Kaggle, NIPS competitions, hackathons
- **Open Source**: Contribute to major ML libraries and frameworks

**Professional Development**
- **Networking**: Conference attendance, research community engagement
- **Mentoring**: Supervise undergraduate students, peer collaboration
- **Teaching**: TA experience, curriculum development, workshop instruction
- **Leadership**: Organize reading groups, student conferences, outreach programs

**📈 Success Metrics and Evaluation**

**Research Impact**
- **Publication Metrics**: Citation count, h-index, impact factor of venues
- **Code Impact**: GitHub stars, downloads, adoption by community
- **Awards**: Best paper awards, fellowships, recognition programs
- **Reproducibility**: Independent validation of results

**Practical Impact**
- **Industry Adoption**: Technology transfer, patent applications
- **Open Source Contributions**: Library adoption, community building
- **Social Impact**: Applications to societal challenges
- **Educational Impact**: Course materials, tutorials, knowledge transfer

The transition from coursework to independent research requires developing both technical depth and research intuition. Success in graduate-level data science depends on combining rigorous methodology with creative problem-solving and effective communication of results to both technical and non-technical audiences.

**🚀 Next Steps for Advanced Study**
1. **Choose Specialization**: Deep dive into specific application domain
2. **Develop Research Proposal**: Identify novel contributions to the field
3. **Build Research Network**: Connect with faculty, industry researchers, peers
4. **Continuous Learning**: Stay current with rapidly evolving field through papers, conferences, and online resources
5. **Practical Experience**: Apply methods to real-world problems through internships or consulting`
    }
  ]
}; 