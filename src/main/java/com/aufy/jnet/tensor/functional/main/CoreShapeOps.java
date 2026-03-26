package com.aufy.jnet.tensor.functional.main;

import java.util.ArrayList;
import java.util.Arrays;

import com.aufy.jnet.stats.Statistics;
import com.aufy.jnet.tensor.core.backend.compute.Engine;
import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.backend.util.ArrayTools;
import com.aufy.jnet.tensor.core.impl.DataContainer;

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
  
  public static DataContainer concat(int axis, DataContainer... tensors) {
    int rankRef = tensors[0].rank;
    int[] shapeRef = tensors[0].shape;
    for (DataContainer tensor : tensors) {
      if (tensor.rank != rankRef) {
        throw new IllegalArgumentException("Mismatching rank for concatenation, attempting to concatenate tensors of rank " + rankRef + " and " + tensor.rank);
      }
      for (int i = 0; i < tensor.rank; i++) {
        if (shapeRef[i] != tensor.shape[i] && i != axis) {
          throw new IllegalArgumentException("Mismatching shape at axis " + i + " for concatenation; attempting to concatenate tensors of shape " + Arrays.toString(shapeRef) + " and " + Arrays.toString(tensor.shape));
        }
      }
      if (axis + 1 > tensor.rank) {
        throw new IllegalArgumentException("Invalid axis " + axis + " for concatenation when one of the tensor has rank " + tensor.rank);
      }
    }

    int[] firstShape = tensors[0].getShape();
    int[] newShape = firstShape.clone();
    
    int totalAxisDim = 0;
    for (DataContainer dc : tensors) totalAxisDim += dc.getShape()[axis];
    newShape[axis] = totalAxisDim;

    double[] newData = new double[ArrayTools.prod(newShape)];
    
    int outerCount = 1;
    for (int i = 0; i < axis; i++) outerCount *= newShape[i];
    
    int innerSize = 1;
    for (int i = axis + 1; i < newShape.length; i++) innerSize *= newShape[i];

    int currentAxisOffset = 0;
    for (DataContainer dc : tensors) {
      double[] srcData = dc.dump();
      int srcAxisDim = dc.getShape()[axis];

      for (int o = 0; o < outerCount; o++) {
        int destPos = (o * totalAxisDim * innerSize) + (currentAxisOffset * innerSize);
        int srcPos = o * srcAxisDim * innerSize;

        System.arraycopy(srcData, srcPos, newData, destPos, srcAxisDim * innerSize);
      }
      currentAxisOffset += srcAxisDim;
    }

    return new DataContainer(newData, newShape);
  }
  
  public static DataContainer stack(int axis, DataContainer... tensors) {
    int[] oldShape = tensors[0].getShape();
    int[] expandedShape = new int[oldShape.length + 1];
    
    for (int i = 0, j = 0; i < expandedShape.length; i++) {
      if (i == axis) {
        expandedShape[i] = tensors.length;
      } else {
        expandedShape[i] = oldShape[j++];
      }
    }

    DataContainer[] expandedTensors = new DataContainer[tensors.length];
    int[] dummyShape = oldShape.clone();
    
    for (int i = 0; i < tensors.length; i++) {
      expandedTensors[i] = new DataContainer(tensors[i].dump(), dummyShape);
    }

    return concat(axis, expandedTensors);
  }

  public static DataContainer slice(DataContainer tensor, int axis) {
    return null;
  }
}
