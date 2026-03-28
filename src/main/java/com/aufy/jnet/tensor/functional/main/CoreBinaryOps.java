package com.aufy.jnet.tensor.functional.main;

import java.util.Arrays;

import com.aufy.jnet.tensor.core.backend.compute.Engine;
import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.backend.func.Binary;
import com.aufy.jnet.tensor.core.impl.RawTensor;

public class CoreBinaryOps {
  /*
  keep this simple because coretensor will implement the rest
   */

  public static RawTensor elementwise(RawTensor a, RawTensor b, Binary operation) {
    if (!Arrays.equals(a.shape, b.shape)) {
      int[] broadcastShapeTarget = Shaping.broadcastedShape(a.shape, b.shape);

      a = CoreShapeOps.broadcast(a, broadcastShapeTarget);
      b = CoreShapeOps.broadcast(b, broadcastShapeTarget);
    }

    double[] resultData = Binary.apply(a.data, b.data, operation);
    return new RawTensor(resultData, a.shape);
  }

  public static RawTensor contract(RawTensor a, RawTensor b, int[] axesA, int[] axesB) {
    int[] resShape = Shaping.calculateResShape(a.shape, b.shape, axesA, axesB);
    double[] resultData = Engine.contract(a.data, a.strides, b.data, b.strides, a.shape, axesA, b.shape, axesB, resShape);

    return new RawTensor(resultData, resShape);
  }
  
}
