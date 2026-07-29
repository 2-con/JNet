package com.aufy.jnet.core.tensor.functional.main;

import com.aufy.jnet.core.backend.interfaces.UnaryOp;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;

public class RawUnaryOps {
  /*
  keep this simple because coretensor will implement the rest
   */
  
  public static RawTensor elementwise(RawTensor tensor, UnaryOp operation) {
    double[] resultData = UnaryOp.apply(tensor.dump(), operation);
    return new RawTensor(resultData, tensor.getShape());
  }
}
