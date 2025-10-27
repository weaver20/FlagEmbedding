import logging
import torch
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod
from transformers.trainer import Trainer

logger = logging.getLogger(__name__)


class AbsRerankerTrainer(ABC, Trainer):
    """
    Abstract class for the trainer of reranker.
    """
    @abstractmethod
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        pass

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.

        Args:
            model (AbsRerankerModel): The model being trained.
            inputs (dict): A dictionary of input tensors to be passed to the model.
            return_outputs (bool, optional): If ``True``, returns both the loss and the model's outputs. Otherwise,
                returns only the loss. Defaults to ``False``.

        Returns:
            Union[torch.Tensor, tuple(torch.Tensor, RerankerOutput)]: The computed loss. If ``return_outputs`` is ``True``,
                also returns the model's outputs in a tuple ``(loss, outputs)``.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on model using inputs.

        Args:
            model: The model to evaluate.
            inputs: The inputs and targets of the model.
            prediction_loss_only: Whether to return only the loss.
            ignore_keys: Keys to ignore when gathering outputs.

        Returns:
            Tuple of (loss, logits, labels) where logits are the reranker scores.
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # Check if we should use sub-batching for memory efficiency
            # Get sub_batch_size from args, default to None (no sub-batching)
            sub_batch_size = getattr(self.args, 'sub_batch_size', None)

            if sub_batch_size is not None and sub_batch_size > 0:
                # Process in sub-batches to save memory during evaluation
                batch_size = inputs['pair']['input_ids'].shape[0]
                all_scores = []
                all_losses = []

                for i in range(0, batch_size, sub_batch_size):
                    end_idx = min(i + sub_batch_size, batch_size)
                    sub_inputs = {
                        'pair': {k: v[i:end_idx] for k, v in inputs['pair'].items()},
                        'teacher_scores': inputs['teacher_scores'][i:end_idx] if inputs.get('teacher_scores') is not None else None
                    }
                    sub_outputs = model(**sub_inputs)
                    all_scores.append(sub_outputs.scores)
                    if sub_outputs.loss is not None:
                        all_losses.append(sub_outputs.loss)

                # Concatenate all sub-batch scores
                logits = torch.cat(all_scores, dim=0)

                # Average losses from sub-batches
                loss = torch.stack(all_losses).mean() if all_losses else None
            else:
                # Standard processing without sub-batching
                outputs = model(**inputs)
                loss = outputs.loss
                logits = outputs.scores

        if prediction_loss_only:
            return (loss, None, None)

        # Return (loss, predictions, labels)
        # For reranker, predictions are scores, labels are implicit (first passage is positive)
        # Create dummy labels - not used by compute_metrics but needed by Trainer
        # Labels are just zeros with same length as logits (one per score)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Detach tensors but keep on GPU for distributed gathering
        # The Trainer will handle moving to CPU after gathering across GPUs
        if logits is not None:
            logits = logits.detach()
        if loss is not None:
            loss = loss.detach()
        if labels is not None:
            labels = labels.detach()

        return (loss, logits, labels)
