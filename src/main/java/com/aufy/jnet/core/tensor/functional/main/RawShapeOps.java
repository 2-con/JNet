package com.aufy.jnet.core.tensor.functional.main;

import com.aufy.jnet.core.tensor.core.backend.compute.Engine;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;

public class RawShapeOps {
  /*
  keep this simple because coretensor will implement the rest
  */

  public static RawTensor permute(RawTensor tensor, int... newOrder) {
    int[] newShape = Shaping.reorder(tensor.getShape(), newOrder);
    int[] newStrides = Shaping.reorder(tensor.getStrides(), newOrder);

    double[] newData = Engine.makeContiguous(tensor.dump(), newShape, newStrides);
    
    return new RawTensor(newData, newShape);
  }
  
  public static RawTensor broadcast(RawTensor tensor, int... targetShape) {
    if (java.util.Arrays.equals(tensor.getShape(), targetShape)) { // dont waste compute
      return tensor;
    }
    
    double[] broadcastedData = Engine.broadcast(tensor.dump(), tensor.getShape(), targetShape);
    return new RawTensor(broadcastedData, targetShape);
  }
  
  public static RawTensor squeeze(RawTensor tensor) {
    return new RawTensor(tensor.dump(), Shaping.squeeze(tensor.getShape()));
  }
  
  public static RawTensor unsqueeze(RawTensor tensor, int... axes) {
    return new RawTensor(tensor.dump(), Shaping.unsqueeze(tensor.getShape(), axes));
  }
  
  public static RawTensor reshape(RawTensor tensor, int... shape) {    
    int[] newShape = Shaping.inferShape(tensor.getShape(), shape);
    
    return new RawTensor(tensor.dump(), newShape);
  }
  
  public static RawTensor concat(int axis, RawTensor... tensors) {
    int[][] shapes = new int[tensors.length][];
    double[][] dataList = new double[tensors.length][];
    int[] resShape = tensors[0].getShape();
    int totalAxisDim = 0;

    for (int i = 0; i < tensors.length; i++) {
      shapes[i] = tensors[i].getShape();
      dataList[i] = tensors[i].dump();
      totalAxisDim += shapes[i][axis];
    }
    resShape[axis] = totalAxisDim;

    double[] resultData = Engine.concat(axis, shapes, dataList, resShape);

    return new RawTensor(resultData, resShape);
  }

  public static RawTensor stack(int axis, RawTensor... tensors) {
    int[] oldShape = tensors[0].getShape();
    int[] resShape = new int[oldShape.length + 1];
    for (int i = 0, j = 0; i < resShape.length; i++) {
      if (i == axis) resShape[i] = tensors.length;
      else resShape[i] = oldShape[j++];
    }

    double[][] dataList = new double[tensors.length][];
    for (int i = 0; i < tensors.length; i++) {
      dataList[i] = tensors[i].dump();
    }

    int[][] shapes = new int[tensors.length][];
    for (int i = 0; i < tensors.length; i++) {
      shapes[i] = tensors[i].getShape();
    }

    double[] resultData = Engine.stack(axis, dataList, shapes, resShape);

    return new RawTensor(resultData, resShape);
  }

  public static RawTensor slice(RawTensor tensor, int axis, int index) {
    int[] newShape = new int[tensor.getRank() - 1];
    for (int i = 0, k = 0; i < tensor.getRank(); i++) {
      if (i != axis) newShape[k++] = tensor.getShape()[i];
    }

    double[] resultData = Engine.slice(tensor.dump(), tensor.getShape(), tensor.getStrides(), axis, index);

    return new RawTensor(resultData, newShape);
  }

}
