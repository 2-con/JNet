package com.aufy.jnet.core.tensor.functional.main;

import java.util.Arrays;

import com.aufy.jnet.core.backend.interfaces.BinaryOp;
import com.aufy.jnet.core.tensor.core.backend.compute.Engine;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;

public class RawBinaryOps {
  /*
  keep this simple because coretensor will implement the rest
  */

  public static RawTensor elementwise(RawTensor a, RawTensor b, BinaryOp operation) {
    if (!Arrays.equals(a.getShape(), b.getShape())) {
      int[] broadcastShapeTarget = Shaping.broadcastedShape(a.getShape(), b.getShape());

      a = RawShapeOps.broadcast(a, broadcastShapeTarget);
      b = RawShapeOps.broadcast(b, broadcastShapeTarget);
    }

    double[] resultData = BinaryOp.apply(a.dump(), b.dump(), operation);
    return new RawTensor(resultData, a.getShape());
  }

  public static RawTensor contract(RawTensor a, RawTensor b, int[] axesA, int[] axesB) {
    int[] resShape = Shaping.calculateResShape(a.getShape(), b.getShape(), axesA, axesB);
    
    double[] resultData = Engine.contract(a.dump(), a.getStrides(), b.dump(), b.getStrides(), a.getShape(), axesA, b.getShape(), axesB, resShape);

    return new RawTensor(resultData, resShape);
  }
  
}
