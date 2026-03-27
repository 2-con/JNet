package com.aufy.jnet.tensor.functional.main;

import com.aufy.jnet.tensor.core.backend.compute.Engine;
import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.impl.RawTensor;

public class CoreShapeOps {
  public static RawTensor permute(RawTensor tensor, int... newOrder) {
    int[] newDims = new int[tensor.shape.length];
    int[] newStrides = new int[tensor.strides.length];
    
    for (int i = 0; i < newOrder.length; i++) {
      newDims[i] = tensor.shape[newOrder[i]];
      newStrides[i] = tensor.strides[newOrder[i]];
    }
    
    return new RawTensor(tensor.data, newDims, newStrides);
  }
  
  public static RawTensor broadcast(RawTensor tensor, int... targetShape) {
    if (java.util.Arrays.equals(tensor.shape, targetShape)) { // dont waste compute
      return tensor;
    }
    
    double[] broadcastedData = Engine.broadcast(tensor.data, tensor.shape, targetShape);
    return new RawTensor(broadcastedData, targetShape);
  }
  
  public static RawTensor squeeze(RawTensor tensor) {
    return new RawTensor(tensor.data, Shaping.squeezeShape(tensor.shape));
  }
  
  public static RawTensor unsqueeze(RawTensor tensor, int... axes) {
    return new RawTensor(tensor.data, Shaping.unsqueezeShape(tensor.shape, axes));
  }
  
  public static RawTensor reshape(RawTensor tensor, int... shape) {    
    int[] newShape = Shaping.inferShape(tensor.shape, shape);
    
    return new RawTensor(tensor.data, newShape);
  }
  
  public static RawTensor concat(int axis, RawTensor... tensors) {
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
    int[] newShape = new int[tensor.rank - 1];
    for (int i = 0, k = 0; i < tensor.rank; i++) {
      if (i != axis) newShape[k++] = tensor.shape[i];
    }

    double[] resultData = Engine.slice(tensor.data, tensor.shape, tensor.strides, axis, index);

    return new RawTensor(resultData, newShape);
  }

}
