// NOTE FOR FUTURE AI:
// When defining lesson sections with code blocks, always use unescaped backticks (`) to open and close the template literal string.
// All triple backticks for code blocks inside the string must be escaped (\`\`\`).
// For example:
//   content: `Some text...\n\`\`\`python\ncode here\n\`\`\``
// This convention avoids accidental termination of the template literal and is the required style for all lessons in this codebase.

import type { LessonData } from "../types";

export const lesson3: LessonData = {
  id: 3,
  title: "Advanced Machine Learning: Theory, Algorithms, and Implementation",
  description: "Comprehensive exploration of machine learning theory, mathematical foundations, optimization algorithms, and state-of-the-art implementations for research and industry applications.",
  sections: [
    {
      title: "Mathematical Foundations and Statistical Learning Theory",
      content: `Machine learning is grounded in statistical learning theory, providing theoretical guarantees for generalization and optimal decision boundaries:

**PAC Learning Framework (Probably Approximately Correct)**
- **Definition**: A learning algorithm is PAC-learnable if it can find a hypothesis h with error ≤ ε with probability ≥ (1-δ)
- **Sample Complexity**: The number of examples needed to achieve PAC-learnability
- **VC Dimension**: Vapnik-Chervonenkis dimension measures hypothesis class complexity
- **Generalization Bounds**: Relate training error to true error via complexity measures

**Information Theory and Learning**
- **Entropy**: H(X) = -∑ p(x) log p(x) measures uncertainty
- **Mutual Information**: I(X;Y) = H(X) - H(X|Y) quantifies information sharing
- **KL Divergence**: D_KL(P||Q) = ∑ p(x) log(p(x)/q(x)) measures distribution distance
- **Maximum Likelihood Estimation**: θ* = argmax_θ ∏ p(x_i|θ)

**Bayesian Learning Theory**
- **Prior/Posterior**: P(θ|D) ∝ P(D|θ)P(θ) via Bayes' theorem
- **MAP Estimation**: θ_MAP = argmax_θ P(θ|D)
- **Bayesian Model Averaging**: Integrates over parameter uncertainty
- **Evidence Approximation**: Marginal likelihood for model selection

The theoretical foundation ensures that ML algorithms have principled guarantees rather than heuristic performance.`
    },
    {
      title: "Advanced Optimization and Gradient Methods",
      content: `Modern machine learning relies on sophisticated optimization algorithms that efficiently navigate high-dimensional parameter spaces:

\`\`\`python interactive
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class OptimizationResult:
    """Container for optimization results and convergence analysis."""
    parameters: np.ndarray
    losses: List[float]
    gradients: List[np.ndarray]
    convergence_rate: float
    iterations: int

class AdvancedOptimizer:
    """Implementation of advanced optimization algorithms with theoretical analysis."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.history = []
    
    def adam_optimizer(self, objective: Callable, gradient: Callable, 
                      x0: np.ndarray, max_iter: int = 1000,
                      beta1: float = 0.9, beta2: float = 0.999, 
                      eps: float = 1e-8) -> OptimizationResult:
        """Adam optimizer with bias correction and convergence analysis."""
        x = x0.copy()
        m = np.zeros_like(x)  # First moment estimate
        v = np.zeros_like(x)  # Second moment estimate
        
        losses = []
        gradients = []
        
        for t in range(1, max_iter + 1):
            # Compute gradient and objective
            grad = gradient(x)
            loss = objective(x)
            
            # Update biased first and second moment estimates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update parameters
            x = x - self.lr * m_hat / (np.sqrt(v_hat) + eps)
            
            losses.append(loss)
            gradients.append(grad.copy())
            
            # Early stopping criterion
            if len(losses) > 10 and abs(losses[-1] - losses[-10]) < 1e-8:
                break
        
        # Compute convergence rate
        if len(losses) > 100:
            convergence_rate = np.mean(np.diff(losses[-50:]))
        else:
            convergence_rate = np.mean(np.diff(losses))
        
        return OptimizationResult(x, losses, gradients, convergence_rate, t)
    
    def natural_gradient(self, objective: Callable, gradient: Callable,
                        fisher_info: Callable, x0: np.ndarray, 
                        max_iter: int = 500) -> OptimizationResult:
        """Natural gradient descent using Fisher Information Matrix."""
        x = x0.copy()
        losses = []
        gradients = []
        
        for t in range(max_iter):
            grad = gradient(x)
            loss = objective(x)
            F = fisher_info(x)  # Fisher Information Matrix
            
            # Natural gradient: F^(-1) * gradient
            try:
                natural_grad = np.linalg.solve(F + 1e-6 * np.eye(len(F)), grad)
            except np.linalg.LinAlgError:
                # Fallback to regular gradient if Fisher matrix is singular
                natural_grad = grad
            
            x = x - self.lr * natural_grad
            losses.append(loss)
            gradients.append(grad.copy())
            
            if len(losses) > 5 and abs(losses[-1] - losses[-5]) < 1e-10:
                break
        
        convergence_rate = np.mean(np.diff(losses)) if len(losses) > 1 else 0
        return OptimizationResult(x, losses, gradients, convergence_rate, t + 1)

# Demonstrate optimization on Rosenbrock function (non-convex optimization)
def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²"""
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """Gradient of Rosenbrock function."""
    dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    dy = 2 * b * (x[1] - x[0]**2)
    return np.array([dx, dy])

def rosenbrock_fisher(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """Approximated Fisher Information Matrix for Rosenbrock."""
    # For demonstration - in practice, this would be problem-specific
    grad = rosenbrock_gradient(x, a, b)
    return np.outer(grad, grad) + 0.1 * np.eye(2)

# Compare optimization algorithms
optimizer = AdvancedOptimizer(learning_rate=0.001)
x0 = np.array([-1.0, 1.0])  # Starting point

print("Optimization Algorithm Comparison on Rosenbrock Function:")
print(f"True minimum: [1.0, 1.0], f_min = 0.0")
print(f"Starting point: {x0}")

# Adam optimization
adam_result = optimizer.adam_optimizer(rosenbrock, rosenbrock_gradient, x0, max_iter=2000)
print(f"\\nAdam Optimizer:")
print(f"Final point: [{adam_result.parameters[0]:.4f}, {adam_result.parameters[1]:.4f}]")
print(f"Final loss: {adam_result.losses[-1]:.8f}")
print(f"Iterations: {adam_result.iterations}")
print(f"Convergence rate: {adam_result.convergence_rate:.8f}")

# Natural gradient optimization
natural_result = optimizer.natural_gradient(rosenbrock, rosenbrock_gradient, 
                                          rosenbrock_fisher, x0, max_iter=1000)
print(f"\\nNatural Gradient:")
print(f"Final point: [{natural_result.parameters[0]:.4f}, {natural_result.parameters[1]:.4f}]")
print(f"Final loss: {natural_result.losses[-1]:.8f}")
print(f"Iterations: {natural_result.iterations}")
print(f"Convergence rate: {natural_result.convergence_rate:.8f}")
\`\`\``
    },
    {
      title: "Probabilistic Models and Bayesian Learning",
      content: `Bayesian approaches provide principled uncertainty quantification and robust learning under limited data:

\`\`\`python interactive
import numpy as np
import scipy.stats as stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from typing import Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

class BayesianLinearRegression:
    """Bayesian Linear Regression with conjugate priors."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize with precision parameters.
        alpha: precision of prior over weights
        beta: precision of noise
        """
        self.alpha = alpha  # Prior precision
        self.beta = beta    # Noise precision
        self.mean = None    # Posterior mean
        self.cov = None     # Posterior covariance
        self.fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """Compute posterior distribution over weights."""
        # Add bias term
        X_design = np.column_stack([np.ones(X.shape[0]), X])
        
        # Posterior covariance: (α*I + β*X^T*X)^(-1)
        S_inv = self.alpha * np.eye(X_design.shape[1]) + self.beta * X_design.T @ X_design
        self.cov = np.linalg.inv(S_inv)
        
        # Posterior mean: β * S * X^T * y
        self.mean = self.beta * self.cov @ X_design.T @ y
        
        self.X_train = X_design
        self.fitted = True
        return self
    
    def predict(self, X_test: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty quantification."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_test_design = np.column_stack([np.ones(X_test.shape[0]), X_test])
        
        # Predictive mean
        y_mean = X_test_design @ self.mean
        
        if return_std:
            # Predictive variance: (1/β + x^T * S * x)
            y_var = np.zeros(X_test.shape[0])
            for i, x in enumerate(X_test_design):
                y_var[i] = 1/self.beta + x.T @ self.cov @ x
            y_std = np.sqrt(y_var)
            return y_mean, y_std
        
        return y_mean
    
    def posterior_samples(self, n_samples: int = 100) -> np.ndarray:
        """Sample from posterior distribution over weights."""
        if not self.fitted:
            raise ValueError("Model must be fitted before sampling")
        
        return np.random.multivariate_normal(self.mean, self.cov, n_samples)

class VariationalInference:
    """Variational Inference for approximate Bayesian learning."""
    
    def __init__(self, model: Callable, prior: Callable):
        self.model = model
        self.prior = prior
        self.variational_params = None
    
    def elbo(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Evidence Lower BOund (ELBO) for variational inference."""
        # Variational parameters: mean and log-variance
        n_params = len(params) // 2
        mu = params[:n_params]
        log_var = params[n_params:]
        
        # Sample from variational distribution
        n_samples = 10
        kl_div = 0
        likelihood = 0
        
        for _ in range(n_samples):
            # Reparameterization trick
            eps = np.random.normal(0, 1, n_params)
            weights = mu + np.exp(0.5 * log_var) * eps
            
            # Likelihood term
            pred = self.model(X, weights)
            likelihood += np.sum(stats.norm.logpdf(y, pred, 0.1))
            
            # KL divergence between variational and prior distributions
            kl_div += np.sum(0.5 * (np.exp(log_var) + mu**2 - 1 - log_var))
        
        return (likelihood / n_samples) - kl_div
    
    def fit(self, X: np.ndarray, y: np.ndarray, n_iter: int = 1000):
        """Optimize variational parameters to maximize ELBO."""
        n_params = X.shape[1] + 1  # Include bias
        
        # Initialize variational parameters
        initial_params = np.concatenate([
            np.random.normal(0, 0.1, n_params),  # mean
            np.random.normal(-2, 0.1, n_params)  # log-variance
        ])
        
        # Optimize ELBO using gradient ascent
        params = initial_params.copy()
        learning_rate = 0.01
        
        for i in range(n_iter):
            # Numerical gradient approximation
            grad = np.zeros_like(params)
            eps = 1e-5
            
            for j in range(len(params)):
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus[j] += eps
                params_minus[j] -= eps
                
                grad[j] = (self.elbo(params_plus, X, y) - self.elbo(params_minus, X, y)) / (2 * eps)
            
            params += learning_rate * grad
            
            if i % 200 == 0:
                elbo_val = self.elbo(params, X, y)
                print(f"Iteration {i}, ELBO: {elbo_val:.4f}")
        
        self.variational_params = params

# Demonstrate Bayesian Learning
print("Bayesian Learning Demonstration:")
print("="*50)

# Generate synthetic data with noise
np.random.seed(42)
X_true = np.linspace(-3, 3, 50).reshape(-1, 1)
y_true = 0.5 * X_true.flatten() + 0.3 * X_true.flatten()**2 + np.random.normal(0, 0.3, 50)

# Split into train/test
train_idx = np.random.choice(50, 30, replace=False)
test_idx = np.setdiff1d(range(50), train_idx)

X_train, y_train = X_true[train_idx], y_true[train_idx]
X_test, y_test = X_true[test_idx], y_true[test_idx]

# Bayesian Linear Regression
print("\\n1. Bayesian Linear Regression:")
blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
blr.fit(X_train, y_train)

# Predictions with uncertainty
y_pred_mean, y_pred_std = blr.predict(X_test, return_std=True)
test_rmse = np.sqrt(np.mean((y_test - y_pred_mean)**2))

print(f"Test RMSE: {test_rmse:.4f}")
print(f"Average prediction uncertainty: {np.mean(y_pred_std):.4f}")
print(f"Posterior weight mean: {blr.mean}")
print(f"Posterior weight std: {np.sqrt(np.diag(blr.cov))}")

# Sample from posterior
weight_samples = blr.posterior_samples(n_samples=5)
print(f"\\nPosterior weight samples:")
for i, sample in enumerate(weight_samples):
    print(f"Sample {i+1}: [{sample[0]:.3f}, {sample[1]:.3f}]")

# Gaussian Process Regression for comparison
print("\\n2. Gaussian Process Regression:")
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
gpr.fit(X_train, y_train)

y_gpr_mean, y_gpr_std = gpr.predict(X_test, return_std=True)
gpr_rmse = np.sqrt(np.mean((y_test - y_gpr_mean)**2))

print(f"GP Test RMSE: {gpr_rmse:.4f}")
print(f"GP Average uncertainty: {np.mean(y_gpr_std):.4f}")
print(f"Optimized kernel: {gpr.kernel_}")
\`\`\``
    },
    {
      title: "Deep Learning Theory and Neural Network Optimization",
      content: `Deep learning combines universal approximation theorems with scalable optimization, enabling learning of complex hierarchical representations:

\`\`\`python interactive
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Optional
import math

class AttentionMechanism(nn.Module):
    """Scaled Dot-Product Attention with theoretical foundations."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        
        return output, attention_weights

class ResidualBlock(nn.Module):
    """Residual connection with layer normalization."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """Apply residual connection: x + dropout(sublayer(norm(x)))"""
        return x + self.dropout(sublayer(self.norm(x)))

class TransformerBlock(nn.Module):
    """Complete Transformer block with attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = AttentionMechanism(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.residual1 = ResidualBlock(d_model, dropout)
        self.residual2 = ResidualBlock(d_model, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        x = self.residual1(x, lambda x: self.attention(x, x, x, mask)[0])
        
        # Feed-forward with residual connection
        x = self.residual2(x, self.feed_forward)
        
        return x

class AdvancedOptimizer:
    """Advanced optimization techniques for deep learning."""
    
    @staticmethod
    def cosine_annealing_scheduler(optimizer: torch.optim.Optimizer, 
                                  T_max: int, eta_min: float = 0) -> torch.optim.lr_scheduler._LRScheduler:
        """Cosine annealing learning rate schedule."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
    
    @staticmethod
    def warmup_scheduler(optimizer: torch.optim.Optimizer, 
                        warmup_steps: int, d_model: int) -> Callable:
        """Learning rate warmup as in 'Attention Is All You Need'."""
        def lr_lambda(step):
            if step == 0:
                step = 1
            return min(step**(-0.5), step * warmup_steps**(-1.5)) * (d_model**(-0.5))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Theoretical Analysis: Universal Approximation
class UniversalApproximator(nn.Module):
    """Demonstrate universal approximation theorem."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Demonstration: Function Approximation
print("Deep Learning Theory Demonstration:")
print("="*50)

# Generate complex function to approximate
def target_function(x):
    """Complex non-linear function for approximation."""
    return np.sin(2 * np.pi * x) * np.exp(-x**2) + 0.5 * np.cos(10 * x)

# Generate training data
X_train = np.random.uniform(-2, 2, 1000).reshape(-1, 1)
y_train = target_function(X_train.flatten()) + np.random.normal(0, 0.05, 1000)

X_test = np.linspace(-2, 2, 200).reshape(-1, 1)
y_test = target_function(X_test.flatten())

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
X_test_tensor = torch.FloatTensor(X_test)

# Create dataset and dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize models with different complexities
models = {
    'Shallow (2 layers)': UniversalApproximator(1, 10, 1, n_layers=2),
    'Deep (5 layers)': UniversalApproximator(1, 50, 1, n_layers=5),
    'Very Deep (10 layers)': UniversalApproximator(1, 100, 1, n_layers=10)
}

print("\\nUniversal Approximation Comparison:")
for name, model in models.items():
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(200):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 50 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"{name} - Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Test performance
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy().flatten()
        mse = np.mean((y_test - y_pred)**2)
        print(f"{name} - Test MSE: {mse:.6f}")

# Demonstrate attention mechanism
print("\\n\\nAttention Mechanism Analysis:")
d_model = 64
seq_length = 10
batch_size = 2

# Create random input sequences
input_seq = torch.randn(batch_size, seq_length, d_model)

# Initialize attention mechanism
attention = AttentionMechanism(d_model, n_heads=8)

# Forward pass
output, attention_weights = attention(input_seq, input_seq, input_seq)

print(f"Input shape: {input_seq.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"Attention weights sum (should be ~1): {attention_weights.sum(dim=-1).mean():.4f}")

# Analyze attention patterns
print("\\nAttention Weight Statistics:")
print(f"Mean attention weight: {attention_weights.mean():.4f}")
print(f"Std attention weight: {attention_weights.std():.4f}")
print(f"Max attention weight: {attention_weights.max():.4f}")
print(f"Min attention weight: {attention_weights.min():.4f}")
\`\`\``
    },
    {
      title: "Advanced Ensemble Methods and Meta-Learning",
      content: `Ensemble methods and meta-learning extend individual models to achieve superior performance through combination and adaptation:

\`\`\`python interactive
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsemble:
    """Advanced ensemble methods with theoretical foundations."""
    
    def __init__(self, base_models: List[Any], meta_model: Any = None):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression()
        self.is_fitted = False
        self.base_predictions = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdvancedEnsemble':
        """Train ensemble using stacking with cross-validation."""
        n_models = len(self.base_models)
        n_samples = X.shape[0]
        
        # Generate out-of-fold predictions for meta-model training
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(self.base_models):
            fold_predictions = np.zeros(n_samples)
            
            for train_idx, val_idx in kfold.split(X):
                # Train base model on fold
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train = y[train_idx]
                
                model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                model_copy.fit(X_fold_train, y_fold_train)
                
                # Predict on validation fold
                if hasattr(model_copy, 'predict_proba'):
                    fold_predictions[val_idx] = model_copy.predict_proba(X_fold_val)[:, 1]
                else:
                    fold_predictions[val_idx] = model_copy.predict(X_fold_val)
            
            meta_features[:, i] = fold_predictions
            
            # Fit base model on full dataset
            self.base_models[i].fit(X, y)
        
        # Train meta-model on out-of-fold predictions
        self.meta_model.fit(meta_features, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get base model predictions
        base_pred = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            if hasattr(model, 'predict_proba'):
                base_pred[:, i] = model.predict_proba(X)[:, 1]
            else:
                base_pred[:, i] = model.predict(X)
        
        # Meta-model final prediction
        return self.meta_model.predict(base_pred)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        base_pred = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            if hasattr(model, 'predict_proba'):
                base_pred[:, i] = model.predict_proba(X)[:, 1]
            else:
                base_pred[:, i] = model.predict(X)
        
        return self.meta_model.predict_proba(base_pred)

class MetaLearner:
    """Model-Agnostic Meta-Learning (MAML) implementation."""
    
    def __init__(self, model_class, alpha: float = 0.01, beta: float = 0.001):
        """
        Initialize meta-learner.
        alpha: inner loop learning rate
        beta: outer loop learning rate
        """
        self.model_class = model_class
        self.alpha = alpha
        self.beta = beta
        self.meta_parameters = None
    
    def inner_update(self, model, support_X: np.ndarray, support_y: np.ndarray, 
                    n_steps: int = 5) -> Any:
        """Perform inner loop adaptation on support set."""
        # This is a simplified version - real MAML requires gradient-based meta-learning
        adapted_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        
        # Simulate gradient-based adaptation
        for _ in range(n_steps):
            adapted_model.fit(support_X, support_y)
            
        return adapted_model
    
    def meta_train(self, task_batches: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], 
                  n_epochs: int = 100):
        """Train meta-learner on multiple tasks."""
        meta_losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            for support_X, support_y, query_X, query_y in task_batches:
                # Initialize base model
                base_model = self.model_class()
                
                # Inner loop: adapt to support set
                adapted_model = self.inner_update(base_model, support_X, support_y)
                
                # Outer loop: evaluate on query set
                query_pred = adapted_model.predict(query_X)
                task_loss = np.mean((query_y - query_pred)**2)  # MSE for regression
                epoch_loss += task_loss
            
            meta_losses.append(epoch_loss / len(task_batches))
            
            if epoch % 20 == 0:
                print(f"Meta-training epoch {epoch}, loss: {meta_losses[-1]:.4f}")
        
        return meta_losses

# Bayesian Model Averaging
class BayesianModelAveraging:
    """Bayesian Model Averaging for model uncertainty."""
    
    def __init__(self, models: List[Any]):
        self.models = models
        self.model_weights = None
        self.log_marginal_likelihoods = None
    
    def compute_model_evidence(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Approximate log marginal likelihood for each model."""
        log_evidences = np.zeros(len(self.models))
        
        for i, model in enumerate(self.models):
            # Use cross-validation as proxy for marginal likelihood
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_log_loss')
            log_evidences[i] = np.mean(cv_scores)
        
        return log_evidences
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianModelAveraging':
        """Compute model weights based on evidence."""
        # Compute log marginal likelihoods
        self.log_marginal_likelihoods = self.compute_model_evidence(X, y)
        
        # Convert to weights using softmax
        max_log_evidence = np.max(self.log_marginal_likelihoods)
        normalized_evidences = self.log_marginal_likelihoods - max_log_evidence
        evidences = np.exp(normalized_evidences)
        self.model_weights = evidences / np.sum(evidences)
        
        # Fit all models
        for model in self.models:
            model.fit(X, y)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Bayesian model averaged predictions."""
        if self.model_weights is None:
            raise ValueError("Model must be fitted first")
        
        weighted_predictions = np.zeros((X.shape[0], 2))  # Binary classification
        
        for i, model in enumerate(self.models):
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                # Convert binary predictions to probabilities
                binary_pred = model.predict(X)
                pred = np.column_stack([1 - binary_pred, binary_pred])
            
            weighted_predictions += self.model_weights[i] * pred
        
        return weighted_predictions

# Demonstration
print("Advanced Ensemble Methods Demonstration:")
print("="*55)

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
n_features = 20

X = np.random.randn(n_samples, n_features)
# Create non-linear decision boundary
true_weights = np.random.randn(n_features)
linear_combination = X @ true_weights
y = (linear_combination + 0.3 * np.sin(3 * linear_combination) + 
     np.random.normal(0, 0.5, n_samples) > 0).astype(int)

# Split data
train_size = int(0.7 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Dataset: {n_samples} samples, {n_features} features")
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Class distribution: {np.bincount(y_train)}")

# Define base models
base_models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    SVC(probability=True, random_state=42),
    LogisticRegression(random_state=42)
]

print("\\n1. Stacking Ensemble:")
ensemble = AdvancedEnsemble(base_models, LogisticRegression())
ensemble.fit(X_train, y_train)

ensemble_pred = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print(f"Stacking ensemble accuracy: {ensemble_accuracy:.4f}")

# Individual model performance for comparison
print("\\nIndividual model performance:")
for i, model in enumerate(base_models):
    model_copy = type(model)(**model.get_params())
    model_copy.fit(X_train, y_train)
    pred = model_copy.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{model.__class__.__name__}: {acc:.4f}")

print("\\n2. Bayesian Model Averaging:")
bma = BayesianModelAveraging([
    RandomForestClassifier(n_estimators=50, random_state=42),
    GradientBoostingClassifier(n_estimators=50, random_state=42),
    LogisticRegression(random_state=42)
])

bma.fit(X_train, y_train)
bma_pred_proba = bma.predict_proba(X_test)
bma_pred = np.argmax(bma_pred_proba, axis=1)
bma_accuracy = accuracy_score(y_test, bma_pred)

print(f"BMA accuracy: {bma_accuracy:.4f}")
print(f"Model weights: {bma.model_weights}")
print(f"Log marginal likelihoods: {bma.log_marginal_likelihoods}")

# Uncertainty quantification
prediction_entropy = -np.sum(bma_pred_proba * np.log(bma_pred_proba + 1e-10), axis=1)
print(f"Average prediction entropy: {np.mean(prediction_entropy):.4f}")
print(f"Uncertainty std: {np.std(prediction_entropy):.4f}")
\`\`\``
    },
    {
      title: "Research Frontiers and Advanced Applications",
      content: `Contemporary machine learning research pushes boundaries in theoretical understanding and practical applications:

**🧠 Theoretical Advances**
- **Geometric Deep Learning**: Extending neural networks to non-Euclidean domains (graphs, manifolds)
- **Causal Inference**: Moving from correlation to causation with do-calculus and causal graphs
- **Federated Learning**: Distributed learning while preserving privacy and data locality
- **Few-Shot Learning**: Learning from minimal examples using meta-learning and transfer

**🔬 Cutting-Edge Research Areas**
- **Neural Architecture Search (NAS)**: Automated neural network design
- **Differentiable Programming**: Making entire programs differentiable for optimization
- **Quantum Machine Learning**: Leveraging quantum computing for ML speedups
- **Neuromorphic Computing**: Brain-inspired hardware for efficient AI

**🏭 Industry Applications and Impact**
- **Scientific Discovery**: Protein folding (AlphaFold), drug discovery, materials science
- **Autonomous Systems**: Self-driving vehicles, robotics, smart manufacturing
- **Creative AI**: GPT models, DALL-E, music generation, code synthesis
- **Climate and Sustainability**: Weather prediction, renewable energy optimization

**🎯 Graduate Research Preparation**
- **Mathematical Rigor**: Linear algebra, probability theory, optimization, statistics
- **Implementation Skills**: PyTorch/TensorFlow, distributed computing, MLOps
- **Research Methodology**: Literature review, experimental design, reproducibility
- **Domain Expertise**: Choose specialization (NLP, computer vision, robotics, etc.)

**📚 Advanced Study Path**
1. **Core Theory**: Study Vapnik's Statistical Learning Theory, Bishop's Pattern Recognition
2. **Specialization**: Choose focus area and dive deep into recent papers
3. **Implementation**: Reproduce key papers and contribute to open-source projects
4. **Research**: Identify open problems and develop novel solutions
5. **Publication**: Submit to conferences (NeurIPS, ICML, ICLR) and journals

**🚀 Career Opportunities**
- **Research Scientist**: Industrial labs (Google AI, OpenAI, DeepMind) or academia
- **ML Engineer**: Production systems, MLOps, scalable inference
- **Data Scientist**: Business applications, analytics, decision support
- **Technical Entrepreneur**: AI startups, product development

**💡 Emerging Paradigms**
- **Foundation Models**: Large pre-trained models adapted for specific tasks
- **Multimodal Learning**: Combining vision, language, and other modalities
- **Continual Learning**: Learning new tasks without forgetting previous ones
- **Explainable AI**: Making AI decisions interpretable and trustworthy

**🔮 Future Directions**
- **Artificial General Intelligence (AGI)**: Moving beyond narrow AI
- **Human-AI Collaboration**: Augmenting rather than replacing human intelligence
- **Ethical AI**: Ensuring fairness, accountability, and societal benefit
- **Green AI**: Reducing computational costs and environmental impact

The field of machine learning continues to evolve rapidly, offering unlimited opportunities for those with strong mathematical foundations, programming skills, and creative problem-solving abilities. Graduate-level study opens doors to cutting-edge research and development of technologies that will shape the future.

**Research Project Ideas for Graduate Students:**
- Develop novel optimization algorithms for large-scale neural networks
- Investigate uncertainty quantification in deep learning models
- Create new architectures for few-shot learning scenarios
- Design federated learning algorithms for heterogeneous data
- Explore the intersection of quantum computing and machine learning
- Build interpretable models for high-stakes applications (healthcare, finance)

Success in graduate-level machine learning requires dedication to continuous learning, strong collaboration skills, and the ability to think both theoretically and practically about complex problems.`
    }
  ]
}; 