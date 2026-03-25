package tensor.functional.main;

import java.util.ArrayList;
import java.util.Arrays;
import stats.Statistics;
import tensor.core.backend.compute.Engine;
import tensor.core.backend.compute.Shaping;
import tensor.core.impl.DataContainer;

public class CoreShapeOps {
  public static DataContainer permute(DataContainer tensor, int... newOrder) {
    ArrayList<Integer> indices = new ArrayList<>();
    
    for (int i = 0; i < newOrder.length; i++) {
      if (newOrder[i] < 0 || newOrder[i] >= tensor.shape.length) {
        throw new IllegalArgumentException("Permutation axes out of bounds");
      }
      
      for (Integer n : indices) {
        if (n == newOrder[i]) {
          throw new IllegalArgumentException("Permutation axes must be unique");
        }
      }
      
      
      indices.add(newOrder[i]);
    }
    
    if (newOrder.length != tensor.shape.length) {
      throw new IllegalArgumentException("Permutation must include all axes.");
    }
    
    int[] newDims = new int[tensor.shape.length];
    int[] newStrides = new int[tensor.strides.length];
    
    for (int i = 0; i < newOrder.length; i++) {
      newDims[i] = tensor.shape[newOrder[i]];
      newStrides[i] = tensor.strides[newOrder[i]];
    }
    
    return new DataContainer(tensor.data, newDims, newStrides);
  }
  
  public static DataContainer broadcast(DataContainer tensor, int... targetShape) {
    if (java.util.Arrays.equals(tensor.shape, targetShape)) {
      return tensor;
    }
    
    // Dimensions must match OR one must be 1
    if (tensor.shape.length != targetShape.length) {
      throw new RuntimeException("Mismatching rank for TensorCore broadcasting, attempting to broadcast " + java.util.Arrays.toString(tensor.shape) + " to " + java.util.Arrays.toString(targetShape));
    }
    
    for (int i = 0; i < tensor.shape.length; i++) {
      if (tensor.shape[i] != targetShape[i] && tensor.shape[i] != 1) {
        throw new RuntimeException("Mismatching dimension at axis " + i + ", attempting to broadcast " + tensor.shape[i] + " to " + targetShape[i]);
      }
    }
    
    double[] broadcastedData = Engine.broadcast(tensor.data, tensor.shape, targetShape);
    return new DataContainer(broadcastedData, targetShape);
  }
  
  public static DataContainer squeeze(DataContainer tensor) {
    return new DataContainer(tensor.data, Shaping.squeezeShape(tensor.shape));
  }
  
  public static DataContainer unsqueeze(DataContainer tensor, int... axes) {
    for (int axis : axes) {
      if (axis < 0 || axis >= tensor.shape.length) {
        throw new IllegalArgumentException("Unsqueeze axes out of bounds: attempted to expand dimensions along axis " + axis);
      }
    }
    
    return new DataContainer(tensor.data, Shaping.unsqueezeShape(tensor.shape, axes));
  }
  
  public static DataContainer reshape(DataContainer tensor, int... shape) {
    if (tensor.size != Statistics.prod(shape)) {
      throw new IllegalArgumentException("Insufficient elements to reshape tensor of shape " + Arrays.toString(tensor.shape) + " (size " + tensor.size + ") into a tensor of shape " + Arrays.toString(shape) + " (size " + tensor.size + ")");
    }
    
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
    
    return new DataContainer(tensor.data, newShape);
  }
  
  // TODO: implement concat (join tensors along an existing dimension)
  public static DataContainer concat(int axis, DataContainer... tensors) {
    System.out.println("Not implemented yet");
    return null;
  }
  
  // TODO: implement stack (join tensors along a new dimension)
  public static DataContainer stack(int axis, DataContainer... tensors) {
    System.out.println("Not implemented yet");
    return null;
  }

}
