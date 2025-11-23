# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import torch
from torchvision.ops.boxes import nms as torchvision_nms
from torchvision.ops import batched_nms
from yacs.config import CfgNode
from typing import List

# The custom op loader and poptorch are no longer needed for standard PyTorch GPU execution.

class CopyTensor(torch.nn.Module):
    def __init__(self, gpu_mode: bool):
        super().__init__()
        self.gpu_mode = gpu_mode
        # No custom op library to load.

    def gpu_copy_tensor(self, input_: torch.Tensor) -> List[torch.Tensor]:
        # This operation is device-agnostic, it works on CPU or GPU.
        return [input_]

    def forward(self, input_: torch.Tensor) -> List[torch.Tensor]:
        # The logic is identical for CPU and GPU, so the flag is less critical here,
        # but we keep it for consistency with the Nms module.
        if self.gpu_mode:
            return self.gpu_copy_tensor(input_)
        else:
            # Fallback to the same operation on the CPU
            return self.gpu_copy_tensor(input_)


class Nms(torch.nn.Module):
    def __init__(self, inference_cfg: CfgNode, gpu_mode: bool):
        super().__init__()
        self.iou_threshold = inference_cfg.iou_threshold
        self.score_threshold = inference_cfg.class_conf_threshold
        self.nms_max_detections = inference_cfg.nms_max_detections
        self.gpu_mode = gpu_mode
        # No custom op library to load.

    def gpu_nms(
        self,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        iou_threshold: float,
        max_detections: int,
    ) -> List[torch.Tensor]:
        """
        Perform non-maximum suppression on predictions on the GPU using a vectorized approach.
        This implementation avoids Python loops over tensor data to prevent DtoH copies.

        Parameters:
            scores (torch.Tensor): BxN tensor of objectness scores per box.
            boxes (torch.Tensor): BxNx4 tensor of boxes (xmin, ymin, xmax, ymax).
                                  These are expected to be "shifted" by class for per-class NMS.
            classes (torch.Tensor): BxN tensor of original class indices per box.
            iou_threshold (float): Predictions overlapping by more than this are discarded.
            max_detections (int): Maximum number of detections per image.

        Returns:
            A list of tensors: [dummy_indices, scores, boxes, classes, num_detections_per_image]
        """
        device = scores.device
        batch_size = scores.shape[0]

        # 1. Flatten all tensors and create a batch index
        scores_flat = scores.flatten()
        boxes_flat = boxes.reshape(-1, 4)
        classes_flat = classes.flatten().long() # Ensure classes are long for indexing

        # Create a batch index tensor to associate each box with its original image
        batch_idxs = torch.arange(batch_size, device=device).view(-1, 1).expand_as(scores).flatten()

        # 2. Filter out boxes with low confidence scores to reduce computation
        score_mask = scores_flat > self.score_threshold
        
        scores_masked = scores_flat[score_mask]
        boxes_masked = boxes_flat[score_mask]
        classes_masked = classes_flat[score_mask]
        batch_idxs_masked = batch_idxs[score_mask]

        # 3. Perform batched NMS across all images in one go
        # The `batch_idxs_masked` ensures NMS is applied independently to each image.
        keep = batched_nms(boxes_masked, scores_masked, batch_idxs_masked, iou_threshold)
        
        # Retrieve the data for the boxes that were kept
        final_scores = scores_masked[keep]
        final_boxes = boxes_masked[keep]
        final_classes = classes_masked[keep]
        final_batch_idxs = batch_idxs_masked[keep]

        # 4. Create padded output tensors
        # Note: The original 'selected_box_indx' is hard to reproduce without original indices
        # and is often not needed. A dummy tensor is returned for API compatibility.
        dummy_indices = torch.full((batch_size, max_detections), -1, dtype=torch.long, device=device)
        output_scores = torch.zeros((batch_size, max_detections), device=device)
        output_boxes = torch.zeros((batch_size, max_detections, 4), device=device)
        output_classes = torch.zeros((batch_size, max_detections), dtype=torch.long, device=device)
        output_num_detections = torch.zeros(batch_size, dtype=torch.int32, device=device)

        # 5. Scatter the kept results back into the padded batch structure
        # This loop iterates over a Python integer range, not tensor data, so it's safe.
        # All operations inside the loop are performed on the GPU.
        for i in range(batch_size):
            item_mask = (final_batch_idxs == i)
            num_detections = item_mask.sum()

            if num_detections == 0:
                continue

            # Limit the number of detections to max_detections
            num_to_keep = min(num_detections, max_detections)
            
            output_num_detections[i] = num_to_keep
            
            # Gather the results for the current batch item
            item_scores = final_scores[item_mask][:num_to_keep]
            item_boxes = final_boxes[item_mask][:num_to_keep]
            item_classes = final_classes[item_mask][:num_to_keep]
            
            # Place the results into the padded output tensors
            output_scores[i, :num_to_keep] = item_scores
            output_boxes[i, :num_to_keep] = item_boxes
            output_classes[i, :num_to_keep] = item_classes

        return [
            dummy_indices,
            output_scores,
            output_boxes,
            output_classes,
            output_num_detections,
        ]

    def forward(
        self, scores: torch.Tensor, boxes: torch.Tensor, classes: torch.Tensor = None
    ) -> List[torch.Tensor]:
        if self.gpu_mode:
            # All tensors (scores, boxes, classes) must be on the GPU before this call.
            return self.gpu_nms(
                scores, boxes, classes, self.iou_threshold, self.nms_max_detections
            )
        else:
            # If not in GPU mode, run the same logic on the CPU.
            # The function is device-agnostic as long as all tensors are on the same device.
             return self.gpu_nms(
                scores, boxes, classes, self.iou_threshold, self.nms_max_detections
            )