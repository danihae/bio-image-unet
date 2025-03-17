import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss.

    Parameters
    ----------
    weight : torch.Tensor, optional
        A manual rescaling weight given to the loss of each batch element.
    size_average : bool, optional
        By default, the losses are averaged over each loss element in the batch.
        If size_average is set to False, the losses are instead summed for each minibatch (default is True).
    """

    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, reduction='mean' if size_average else 'sum')

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
        return self.bce_loss(logits, targets)


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
        probs = torch.sigmoid(logits)
        num = targets.size(0)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2).sum(1)
        score = 2. * (intersection + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        return 1 - score.mean()


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
        self.bce = BCELoss()
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


class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, predictions):
        """
        predictions: Tensor of shape (B, C, Z, X, Y)
                     where Z is the temporal dimension
        """
        # Compute loss between consecutive frames along the Z dimension
        loss = self.loss_fn(predictions[:, :, 1:, :, :], predictions[:, :, :-1, :, :])

        return loss


class BCEDiceTemporalLoss(nn.Module):
    def __init__(self, loss_params=(1.0, 0.1)):
        """
        Combined loss for 3D/temporal segmentation with BCEDice and temporal consistency.

        Args:
            loss_params (tuple): Weights for (BCEDice, TemporalConsistency) losses.
                                Default is (1.0, 0.1).
        """
        super(BCEDiceTemporalLoss, self).__init__()
        self.bce_dice_loss = BCEDiceLoss(1, 1)  # Your existing implementation
        self.temporal_consistency_loss = TemporalConsistencyLoss()
        self.loss_params = loss_params

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (batch_size, frames, channels, height, width)
            targets: Tensor of shape (batch_size, frames, channels, height, width)
        """
        # Compute BCEDice loss using your implementation
        bce_dice_loss_value = self.bce_dice_loss(predictions, targets)

        # Compute Temporal Consistency Loss
        temporal_consistency_loss_value = self.temporal_consistency_loss(predictions)

        # Combine losses with weights from the tuple
        total_loss = (
                self.loss_params[0] * bce_dice_loss_value +
                self.loss_params[1] * temporal_consistency_loss_value
        )

        return total_loss
