package com.aufy.jnet.tensor.graph.main;

import java.util.List;

import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.impl.DataContainer;
import com.aufy.jnet.tensor.core.impl.TensorCore; 
import com.aufy.jnet.tensor.functional.main.CoreShapeOps;

public class ShapeOps {
  public static TensorCore permute(TensorCore tensor, int... axes) {
    TensorCore out = new TensorCore(CoreShapeOps.permute(tensor.core, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      int[] inverse = new int[axes.length];

      for (int i = 0; i < axes.length; i++) {
        inverse[axes[i]] = i;
      }

      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        DataContainer gradInputCore = CoreShapeOps.permute(grad.core, inverse);
        tensor.accumulate(new TensorCore(gradInputCore));
      };
    }

    return out;
  }

  public static TensorCore reshape(TensorCore tensor, int... shape) {
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
    TensorCore out = new TensorCore(CoreShapeOps.reshape(tensor.core, newShape));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        DataContainer gradInputCore = CoreShapeOps.reshape(grad.core, tensor.shape);
        tensor.accumulate(new TensorCore(gradInputCore));
      };
    }

    return out;
  }

  public static TensorCore squeeze(TensorCore tensor) {
    TensorCore out = new TensorCore(CoreShapeOps.squeeze(tensor.core));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);
      
      out.derivative = (grad) -> {
        DataContainer reshaped = CoreShapeOps.reshape(grad.core, tensor.shape);
        tensor.accumulate(new TensorCore(reshaped));
      };
    }
    
    return out;
  }
  
  public static TensorCore unsqueeze(TensorCore tensor) {
    TensorCore out = new TensorCore(CoreShapeOps.unsqueeze(tensor.core));
    out.requiresGrad = tensor.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        DataContainer reshaped = CoreShapeOps.reshape(grad.core, tensor.shape);
        tensor.accumulate(new TensorCore(reshaped));
      };
    }

    return out;
  }

  public static TensorCore concat(int axis, TensorCore... tensors) {
    DataContainer[] tensorsData = new DataContainer[tensors.length];
    boolean requiresGrad = false;

    for (int i = 0; i < tensors.length; i++) {
      tensorsData[i] = tensors[i].core;
      if (tensors[i].requiresGrad) requiresGrad = true;
    }

    TensorCore out = new TensorCore(CoreShapeOps.concat(axis, tensorsData));
    out.requiresGrad = requiresGrad;

    if (requiresGrad) {
      out.parents = List.of(tensors);
      
      out.derivative = (grad) -> {
        // TODO: actualy impliment the logic
      };
    }

    return out;
  }

  public static TensorCore stack(int axis, TensorCore... tensors) {
    DataContainer[] tensorsData = new DataContainer[tensors.length];
    boolean requiresGrad = false;

    for (int i = 0; i < tensors.length; i++) {
      tensorsData[i] = tensors[i].core;
      if (tensors[i].requiresGrad) requiresGrad = true;
    }

    TensorCore out = new TensorCore(CoreShapeOps.stack(axis, tensorsData));
    out.requiresGrad = requiresGrad;

    if (requiresGrad) {
      out.parents = List.of(tensors);
      
      out.derivative = (grad) -> {
        // TODO: actualy impliment the logic
      };
    }

    return out;
  }

}
