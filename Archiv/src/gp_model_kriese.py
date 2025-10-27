import torch
import gpytorch
from sklearn.metrics import r2_score, mean_absolute_error
from itertools import chain


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type="RBF"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel_type == "RBF":
            base_kernel = gpytorch.kernels.RBFKernel()
        elif kernel_type == "Matern":
            base_kernel = gpytorch.kernels.MaternKernel()
        elif kernel_type == "Linear":
            base_kernel = gpytorch.kernels.LinearKernel()
        else:
            raise ValueError(f"Unbekannter Kernel: {kernel_type}")

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def log_hyperparameters(model, likelihood, logger=None, prefix="Final"):
    """Loggt die wichtigsten Hyperparameter des GP-Modells."""
    try:
        noise = likelihood.noise.item()
        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        outputscale = model.covar_module.outputscale.item()
        mean_const = model.mean_module.constant.item()

        msg = (
            f"{prefix} Hyperparameter: "
            f"noise={noise:.4f}, lengthscale={lengthscale:.4f}, "
            f"outputscale={outputscale:.4f}, mean={mean_const:.4f}"
        )
        if logger:
            logger.info(msg)
        else:
            print("[INFO]", msg)
    except Exception as e:
        if logger:
            logger.warning(f"Konnte Hyperparameter nicht loggen: {e}")
        else:
            print(f"[WARN] Konnte Hyperparameter nicht loggen: {e}")


def train_gp_model(X_train, y_train, X_val, y_val, settings, logger=None):
    # Hyperparameter aus Settings
    hp = settings.training.gp
    lr = hp.learning_rate
    training_iter = hp.training_iter
    kernel_type = hp.kernel
    init_params = hp.init_params.as_dict()

    # Torch-Tensoren nur erzeugen, wenn noch keine
    if not isinstance(X_train, torch.Tensor):
        train_x = torch.tensor(X_train, dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.float32)
        val_x   = torch.tensor(X_val, dtype=torch.float32)
        val_y   = torch.tensor(y_val, dtype=torch.float32)
    else:
        train_x, train_y, val_x, val_y = X_train, y_train, X_val, y_val
        

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood, kernel_type=kernel_type)
    model.initialize(**init_params)

    if logger:
        log_hyperparameters(model, likelihood, logger, prefix="Initial")

    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam(
        #list(model.parameters()) + list(likelihood.parameters()),
        #lr=lr
        chain(model.likelihood.parameters(), model.covar_module.parameters()),
        lr=lr
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(val_x))
        mean = preds.mean.detach().cpu().numpy()  # .cpu() sicherheitshalber

    r2 = r2_score(val_y.cpu().numpy() if isinstance(val_y, torch.Tensor) else val_y, mean)
    mae = mean_absolute_error(val_y.cpu().numpy() if isinstance(val_y, torch.Tensor) else val_y, mean)

    if logger:
        log_hyperparameters(model, likelihood, logger, prefix="Final")

    return model, {"r2": r2, "mae": mae}
