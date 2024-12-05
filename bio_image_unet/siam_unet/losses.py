import torch
from torch import nn as nn


class BCEDiceLoss(nn.Module):
    """
    Combines Binary Cross-Entropy Loss and Dice Loss.

    Parameters
    ----------
    alpha : float
        Weight for the Binary Cross-Entropy Loss component.
    beta : float
        Weight for the Dice Loss component.
    """
    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        """
        Forward pass.

        Parameters
        ----------
        logits : torch.Tensor
            The predicted outputs.
        targets : torch.Tensor
            The ground truth labels.

        Returns
        -------
        torch.Tensor
            The combined loss value.
        """
        return self.alpha * self.bce(logits, targets) + self.beta * self.dice(logits, targets)


class logcoshDiceLoss(nn.Module):
    """
    Log-Cosh Dice Loss.

    Combines the Log-Cosh function with Dice Loss for improved stability.
    """
    def __init__(self):
        super(logcoshDiceLoss, self).__init__()
        self.dice = SoftDiceLoss()

    def forward(self, logits, targets):
        """
        Forward pass.

        Parameters
        ----------
        logits : torch.Tensor
            The predicted outputs.
        targets : torch.Tensor
            The ground truth labels.

        Returns
        -------
        torch.Tensor
            The Log-Cosh Dice loss value.
        """
        x = self.dice(logits, targets)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2)


class BCELoss2d(nn.Module):
    """
    Binary Cross-Entropy Loss for 2D data.

    Parameters
    ----------
    weight : torch.Tensor, optional
        A manual rescaling weight given to the loss of each batch element.
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """
    def __init__(self, weight=None, reduction='mean', **kwargs):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, reduction=reduction)

    def forward(self, logits, targets):
        """
        Forward pass.

        Parameters
        ----------
        logits : torch.Tensor
            The predicted outputs.
        targets : torch.Tensor
            The ground truth labels.

        Returns
        -------
        torch.Tensor
            The binary cross-entropy loss value.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class weightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.

    Parameters
    ----------
    alpha : float, optional
        Weight for positive examples (default is 1).
    beta : float, optional
        Weight for negative examples (default is 0.1).
    """
    def __init__(self, alpha=1, beta=0.1):
        super(weightedBCELoss, self).__init__()
        self.bce = nn.BCELoss(reduce=False)
        self.alpha, self.beta = alpha, beta

    def forward(self, logits, targets):
        """
        Forward pass.

        Parameters
        ----------
        logits : torch.Tensor
            The predicted outputs.
        targets : torch.Tensor
            The ground truth labels.

        Returns
        -------
        torch.Tensor
            The weighted binary cross-entropy loss value.
        """
        probs = torch.sigmoid(logits)
        # compute weights
        weights = torch.clone(targets)
        weights[targets >= 0.5] = self.alpha
        weights[targets < 0.5] = self.beta
        # compute loss
        loss = torch.mean(self.bce(probs, targets) * weights)
        return loss


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss.

    Parameters
    ----------
    smooth : float, optional
        Smoothing factor to avoid division by zero (default is 1.0).
    """
    def __init__(self, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Forward pass.

        Parameters
        ----------
        logits : torch.Tensor
            The predicted outputs.
        targets : torch.Tensor
            The ground truth labels.

        Returns
        -------
        torch.Tensor
            The soft dice loss value.
        """
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


class TverskyLoss(nn.Module):
    """
    Tversky Loss.

    Parameters
    ----------
    alpha : float, optional
        Weight of false positives (default is 0.5).
    beta : float, optional
        Weight of false negatives (default is 0.5).
    smooth : float, optional
        Smoothing factor to avoid division by zero (default is 1).
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            The predicted outputs.
        targets : torch.Tensor
            The ground truth labels.

        Returns
        -------
        torch.Tensor
            The Tversky loss value.
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - Tversky


class logcoshTverskyLoss(nn.Module):
    """
    Log-Cosh Tversky Loss.

    Parameters
    ----------
    alpha : float, optional
        Weight of false positives (default is 0.5).
    beta : float, optional
        Weight of false negatives (default is 0.5).
    smooth : float, optional
        Smoothing factor to avoid division by zero (default is 1).
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(logcoshTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            The predicted outputs.
        targets : torch.Tensor
            The ground truth labels.

        Returns
        -------
        torch.Tensor
            The log-cosh Tversky loss value.
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return torch.log(torch.cosh(1 - Tversky))