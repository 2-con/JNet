package com.aufy.jnet.tensor.graph.main;

import java.util.List;

import com.aufy.jnet.tensor.core.backend.compute.Engine;
import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.impl.CoreTensor;
import com.aufy.jnet.tensor.core.impl.RawTensor;
import com.aufy.jnet.tensor.functional.main.CoreShapeOps;

public class ShapeOps {
  public static CoreTensor permute(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(CoreShapeOps.permute(tensor.core, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      int[] inverse = new int[axes.length];

      for (int i = 0; i < axes.length; i++) {
        inverse[axes[i]] = i;
      }

      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor gradInputCore = CoreShapeOps.permute(grad.core, inverse);
        tensor.accumulate(new CoreTensor(gradInputCore));
      };
    }

    return out;
  }

  public static CoreTensor reshape(CoreTensor tensor, int... shape) {
    int countNegatives = 0;
    for (int n : shape) {
      if (n < -1) {
        throw new IllegalArgumentException("Dimension inference only works on -1, got " + n + " instead");
      }
      if (n == -1) {
        countNegatives++;
      }
      if (countNegatives > 1) {
        throw new IllegalArgumentException("Only one dimension can be inferred, got " + countNegatives + " dimensions to infer");
      }
    }

    int[] newShape = Shaping.inferShape(tensor.shape, shape);
    CoreTensor out = new CoreTensor(CoreShapeOps.reshape(tensor.core, newShape));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor gradInputCore = CoreShapeOps.reshape(grad.core, tensor.shape);
        tensor.accumulate(new CoreTensor(gradInputCore));
      };
    }

    return out;
  }

  public static CoreTensor squeeze(CoreTensor tensor) {
    CoreTensor out = new CoreTensor(CoreShapeOps.squeeze(tensor.core));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);
      
      out.derivative = (grad) -> {
        RawTensor reshaped = CoreShapeOps.reshape(grad.core, tensor.shape);
        tensor.accumulate(new CoreTensor(reshaped));
      };
    }
    
    return out;
  }
  
  public static CoreTensor unsqueeze(CoreTensor tensor) {
    CoreTensor out = new CoreTensor(CoreShapeOps.unsqueeze(tensor.core));
    out.requiresGrad = tensor.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor reshaped = CoreShapeOps.reshape(grad.core, tensor.shape);
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

    CoreTensor out = new CoreTensor(CoreShapeOps.concat(axis, tensorsData));
    out.requiresGrad = requiresGrad;

    if (requiresGrad) {
      out.parents = List.of(tensors);
      
      out.derivative = (grad) -> {
        int offset = 0;
        for (CoreTensor parent : tensors) {
          int dimSize = parent.core.shape[axis];

          double[] partialGrad = Engine.rangedSlice(grad.dump(), grad.shape, grad.core.strides, axis, offset, offset + dimSize);

          parent.grad = BinaryOps.add(parent.grad, new CoreTensor(partialGrad, parent.core.getShape()));
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

    CoreTensor out = new CoreTensor(CoreShapeOps.stack(axis, tensorsData));
    out.requiresGrad = requiresGrad;

    if (requiresGrad) {
      out.parents = List.of(tensors);
      
      out.derivative = (grad) -> {
        for (int i = 0; i < tensors.length; i++) {
          // Since stack added a dimension, a single-index slice returns the original shape
          double[] partialGrad = Engine.slice(grad.dump(), grad.shape, grad.core.strides, axis, i);
          
          tensors[i].grad = BinaryOps.add(tensors[i].grad, new CoreTensor(partialGrad, tensors[i].core.getShape()));
        }
      };
    }

    return out;
  }
  
  public static CoreTensor slice(CoreTensor tensor, int axis, int index) {
    CoreTensor out = new CoreTensor(CoreShapeOps.slice(tensor.core, axis, index));
    out.requiresGrad = tensor.requiresGrad;
  
    if (out.requiresGrad) {
      out.parents = List.of(tensor);
      
      out.derivative = (grad) -> {
        // Gradient is smaller than input. Place it in a zero-filled array of original shape.
        double[] expandedGradData = new double[tensor.core.getSize()];

        Engine.place(grad.dump(), expandedGradData, tensor.core.getShape(), tensor.core.getStrides(), axis, index);
        tensor.grad = BinaryOps.add(tensor.grad, new CoreTensor(expandedGradData, tensor.core.getShape()));
      };
    }
  
    return out;
  }
}
