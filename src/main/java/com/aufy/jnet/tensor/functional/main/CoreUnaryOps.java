package com.aufy.jnet.tensor.functional.main;

import com.aufy.jnet.tensor.core.backend.func.Unary;
import com.aufy.jnet.tensor.core.impl.RawTensor;

public class CoreUnaryOps {
  /*
  keep this simple because coretensor will implement the rest
   */
  
  public static RawTensor apply(RawTensor tensor, Unary operation) {
    double[] resultData = Unary.apply(tensor.data, operation);
    return new RawTensor(resultData, tensor.shape);
  }

}
