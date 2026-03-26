package com.aufy.jnet.tensor.functional.main;

import java.util.ArrayList;
import java.util.Arrays;

import com.aufy.jnet.stats.primitive.Statistics;
import com.aufy.jnet.tensor.core.backend.compute.Engine;
import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.impl.RawTensor;

public class CoreShapeOps {
  public static RawTensor permute(RawTensor tensor, int... newOrder) {
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
    
    return new RawTensor(tensor.data, newDims, newStrides);
  }
  
  public static RawTensor broadcast(RawTensor tensor, int... targetShape) {
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
    return new RawTensor(broadcastedData, targetShape);
  }
  
  public static RawTensor squeeze(RawTensor tensor) {
    return new RawTensor(tensor.data, Shaping.squeezeShape(tensor.shape));
  }
  
  public static RawTensor unsqueeze(RawTensor tensor, int... axes) {
    for (int axis : axes) {
      if (axis < 0 || axis >= tensor.shape.length) {
        throw new IllegalArgumentException("Unsqueeze axes out of bounds: attempted to expand dimensions along axis " + axis);
      }
    }
    
    return new RawTensor(tensor.data, Shaping.unsqueezeShape(tensor.shape, axes));
  }
  
  public static RawTensor reshape(RawTensor tensor, int... shape) {
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
    
    return new RawTensor(tensor.data, newShape);
  }
  
  public static RawTensor concat(int axis, RawTensor... tensors) {
    int rankRef = tensors[0].rank;
    int[] shapeRef = tensors[0].shape;
    for (RawTensor tensor : tensors) {
      if (tensor.rank != rankRef) throw new IllegalArgumentException("Mismatching tensor rank for concatenation when the first tensor listed for concatenation has a rank of " + rankRef);
      if (axis >= tensor.rank) throw new IllegalArgumentException("Concatenation axis out of bounds; tensor of rank " + tensor.rank + " cannot be concatenated along axis " + axis);

      for (int i = 0; i < rankRef; i++) if (i != axis && shapeRef[i] != tensor.shape[i]) throw new IllegalArgumentException("Mismatching dimension at axis " + i + " when concatenating tensors of shape " + Arrays.toString(shapeRef) + " and " + Arrays.toString(tensor.shape));
    }

    int[][] shapes = new int[tensors.length][];
    double[][] dataList = new double[tensors.length][];
    int[] resShape = tensors[0].shape.clone();
    int totalAxisDim = 0;

    for (int i = 0; i < tensors.length; i++) {
      shapes[i] = tensors[i].shape;
      dataList[i] = tensors[i].dump();
      totalAxisDim += shapes[i][axis];
    }
    resShape[axis] = totalAxisDim;

    double[] resultData = Engine.concat(axis, shapes, dataList, resShape);

    return new RawTensor(resultData, resShape);
  }

  public static RawTensor stack(int axis, RawTensor... tensors) {
    int rankRef = tensors[0].rank;
    int[] refShape = tensors[0].shape;
    for (RawTensor tensor : tensors) {
      if (tensor.rank != rankRef) throw new IllegalArgumentException("Mismatching tensor rank for concatenation when the first tensor listed for concatenation has a rank of" + rankRef);

      for (int i = 0; i < rankRef; i++) if (tensor.shape[i] != refShape[i]) throw new IllegalArgumentException("Mismatching dimension for stacking tensors: all tensors must have the same shape");
    }

    int[] oldShape = tensors[0].shape;
    int[] resShape = new int[oldShape.length + 1];
    for (int i = 0, j = 0; i < resShape.length; i++) {
      if (i == axis) resShape[i] = tensors.length;
      else resShape[i] = oldShape[j++];
    }

    double[][] dataList = new double[tensors.length][];
    for (int i = 0; i < tensors.length; i++) {
      dataList[i] = tensors[i].dump();
    }

    double[] resultData = Engine.stack(axis, dataList, resShape);

    return new RawTensor(resultData, resShape);
  }

  public static RawTensor slice(RawTensor tensor, int axis, int index) {
    if (index < 0 || index >= tensor.shape[axis]) {
      throw new IndexOutOfBoundsException("Invalid index to slice tensor along axis " + axis + ": " + index + " out of bounds for axis ");
    }

    int[] newShape = new int[tensor.rank - 1];
    for (int i = 0, k = 0; i < tensor.rank; i++) {
      if (i != axis) newShape[k++] = tensor.shape[i];
    }

    double[] resultData = Engine.slice(tensor.data, tensor.shape, tensor.strides, axis, index);

    return new RawTensor(resultData, newShape);
  }

}
