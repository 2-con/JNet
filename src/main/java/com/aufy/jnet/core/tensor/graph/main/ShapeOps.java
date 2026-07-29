package com.aufy.jnet.core.tensor.graph.main;

import java.util.List;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.tensor.core.backend.compute.Engine;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;
import com.aufy.jnet.core.tensor.functional.main.RawReductionOps;
import com.aufy.jnet.core.tensor.functional.main.RawShapeOps;

public class ShapeOps {
  /*
  dont go overboard with the additions, Tensor will do the job. just add the core stuff and Tensor will do the rest
  */
  
  public static CoreTensor permute(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(RawShapeOps.permute(tensor.core, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      int[] inverse = new int[axes.length];

      for (int i = 0; i < axes.length; i++) {
        inverse[axes[i]] = i;
      }

      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor gradInputCore = RawShapeOps.permute(grad.core, inverse);
        tensor.accumulate(new CoreTensor(gradInputCore));
      };
    }

    return out;
  }

  public static CoreTensor broadcast(CoreTensor tensor, int... targetShape) {
    if (java.util.Arrays.equals(tensor.shape, targetShape)) { // dont waste compute
      return tensor;
    }
    
    CoreTensor out = new CoreTensor(Engine.broadcast(tensor.dump(), tensor.shape, targetShape), targetShape);
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        int[] axesExpanded = Tools.findIndices(tensor.shape, 1);

        RawTensor gradInputCore = RawReductionOps.reduce(grad.core, Reductions::sum, axesExpanded);
        tensor.accumulate(new CoreTensor(gradInputCore));
      };
    }

    return out;
  }

  public static CoreTensor reshape(CoreTensor tensor, int... shape) {
    int[] newShape = Shaping.inferShape(tensor.shape, shape);
    CoreTensor out = new CoreTensor(RawShapeOps.reshape(tensor.core, newShape));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor gradInputCore = RawShapeOps.reshape(grad.core, tensor.shape);
        tensor.accumulate(new CoreTensor(gradInputCore));
      };
    }

    return out;
  }

  public static CoreTensor squeeze(CoreTensor tensor) {
    CoreTensor out = new CoreTensor(RawShapeOps.squeeze(tensor.core));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);
      
      out.derivative = (grad) -> {
        RawTensor reshaped = RawShapeOps.reshape(grad.core, tensor.shape);
        tensor.accumulate(new CoreTensor(reshaped));
      };
    }
    
    return out;
  }
  
  public static CoreTensor unsqueeze(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(RawShapeOps.unsqueeze(tensor.core, axes));
    out.requiresGrad = tensor.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor reshaped = RawShapeOps.reshape(grad.core, tensor.shape);
        tensor.accumulate(new CoreTensor(reshaped));
      };
    }

    return out;
  }

  public static CoreTensor concat(int axis, CoreTensor... tensors) {
    RawTensor[] tensorsData = new RawTensor[tensors.length];
    boolean requiresGrad = false;

    for (int i = 0; i < tensors.length; i++) {
      tensorsData[i] = tensors[i].core;
      if (tensors[i].requiresGrad) requiresGrad = true;
    }

    CoreTensor out = new CoreTensor(RawShapeOps.concat(axis, tensorsData));
    out.requiresGrad = requiresGrad;

    if (requiresGrad) {
      out.parents = List.of(tensors);
      
      out.derivative = (grad) -> {
        int offset = 0;
        for (CoreTensor parent : tensors) {
          int dimSize = parent.shape[axis];

          double[] partialGrad = Engine.rangedSlice(grad.dump(), grad.shape, grad.core.getStrides(), axis, offset, offset + dimSize);

          parent.grad = BinaryOps.add(parent.grad, new CoreTensor(partialGrad, parent.shape));
          offset += dimSize;
        }
      };
    }

    return out;
  }

  public static CoreTensor stack(int axis, CoreTensor... tensors) {
    RawTensor[] tensorsData = new RawTensor[tensors.length];
    boolean requiresGrad = false;

    for (int i = 0; i < tensors.length; i++) {
      tensorsData[i] = tensors[i].core;
      if (tensors[i].requiresGrad) requiresGrad = true;
    }

    CoreTensor out = new CoreTensor(RawShapeOps.stack(axis, tensorsData));
    out.requiresGrad = requiresGrad;

    if (requiresGrad) {
      out.parents = List.of(tensors);
      
      out.derivative = (grad) -> {
        for (int i = 0; i < tensors.length; i++) {
          // Since stack added a dimension, a single-index slice returns the original shape
          double[] partialGrad = Engine.slice(grad.dump(), grad.shape, grad.core.getStrides(), axis, i);
          
          tensors[i].grad = BinaryOps.add(tensors[i].grad, new CoreTensor(partialGrad, tensors[i].core.getShape()));
        }
      };
    }

    return out;
  }
  
  public static CoreTensor slice(CoreTensor tensor, int axis, int index) {
    CoreTensor out = new CoreTensor(RawShapeOps.slice(tensor.core, axis, index));
    out.requiresGrad = tensor.requiresGrad;
  
    if (out.requiresGrad) {
      out.parents = List.of(tensor);
      
      out.derivative = (grad) -> {
        // Gradient is smaller than input. Place it in a zero-filled array of original shape.
        double[] expandedGradData = Engine.insert(grad.dump(), tensor.core.getShape(), tensor.core.getStrides(), axis, index);
        tensor.grad = BinaryOps.add(tensor.grad, new CoreTensor(expandedGradData, tensor.core.getShape()));
      };
    }
  
    return out;
  }
}
