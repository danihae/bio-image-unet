import torch
import torch.nn as nn
import torch.nn.functional as F


# Classification loss functions

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()  # Use BCELoss for probabilities

    def forward(self, inputs, targets):
        assert torch.all((inputs >= 0) & (inputs <= 1)), "Inputs must be between 0 and 1"
        assert torch.all((targets >= 0) & (targets <= 1)), "Targets must be between 0 and 1"

        # Compute BCE loss with probabilities
        bce_loss = self.bce(inputs, targets)

        # Dice loss calculation
        smooth = 1e-5
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # Combine BCE and Dice losses
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - Tversky


class logcoshTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(logcoshTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return torch.log(torch.cosh(1 - Tversky))


# Regression loss functions

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inputs, targets):
        return ((inputs - targets) ** 2).mean()


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.abs(inputs - targets).mean()


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, inputs, targets):
        diff = torch.abs(inputs - targets)
        loss = torch.where(diff < self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta))
        return loss.mean()


def gradient_loss(pred, target):
    """Calculate gradient loss comparing spatial derivatives of pred and target."""
    # Calculate gradients in y and x directions
    dy_true, dx_true = torch.gradient(target, dim=(-2, -1))  # dim=-2 is y direction, dim=-1 is x direction
    dy_pred, dx_pred = torch.gradient(pred, dim=(-2, -1))

    # Calculate MSE between gradients
    gradient_loss_y = F.mse_loss(dy_pred, dy_true)
    gradient_loss_x = F.mse_loss(dx_pred, dx_true)

    return gradient_loss_y + gradient_loss_x


class DistanceGradientLoss(torch.nn.Module):
    """Combined loss for distance regression with gradient preservation."""

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # Distance loss (MSE)
        distance_loss = F.mse_loss(pred, target)

        # Gradient loss
        grad_loss = gradient_loss(pred, target)

        # Combined loss
        total_loss = distance_loss + self.alpha * grad_loss

        return total_loss


class WeightedDistanceGradientLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha  # gradient loss weight
        self.beta = beta  # weight for non-zero regions

    def forward(self, pred, target):
        # Create weight mask for non-zero regions
        weights = torch.where(target > 0, self.beta, 1.0 - self.beta)

        # Weighted distance loss (combining MSE and MAE for robustness)
        mse_loss = F.mse_loss(pred * weights, target * weights, reduction='mean')
        mae_loss = F.l1_loss(pred * weights, target * weights, reduction='mean')
        distance_loss = mse_loss + mae_loss

        # Weighted gradient loss
        grad_loss = gradient_loss(pred * weights, target * weights)

        return distance_loss + self.alpha * grad_loss


class WeightedVectorFieldLoss(nn.Module):
    def __init__(self, beta=0.5, magnitude_weight=0.3):
        super().__init__()
        self.beta = beta
        self.magnitude_weight = magnitude_weight

    def forward(self, pred_vectors, true_vectors):
        """
        Args:
            pred_vectors: Predicted vector field (B, 2, H, W)
            true_vectors: Ground truth vector field (B, 2, H, W)
        """
        # Create mask where vectors are valid (both components non-zero)
        mask = ~((true_vectors[:, 0] == 0) & (true_vectors[:, 1] == 0))

        # Create weights
        weights = torch.where(mask, self.beta, 1.0 - self.beta)

        # Component-wise losses
        mse_loss = F.mse_loss(pred_vectors * weights[:, None],
                              true_vectors * weights[:, None],
                              reduction='mean')
        mae_loss = F.l1_loss(pred_vectors * weights[:, None],
                             true_vectors * weights[:, None],
                             reduction='mean')

        # Magnitude loss
        pred_magnitude = torch.sum(pred_vectors ** 2, dim=1)
        true_magnitude = torch.sum(true_vectors ** 2, dim=1)
        magnitude_loss = F.mse_loss(pred_magnitude * weights,
                                    true_magnitude * weights,
                                    reduction='mean')

        return mse_loss + mae_loss + self.magnitude_weight * magnitude_loss
