package com.aufy.jnet.tensor.functional.main;

import com.aufy.jnet.tensor.core.backend.func.Unary;
import com.aufy.jnet.tensor.core.impl.DataContainer;

public class CoreUnaryOps {
  public static DataContainer apply(DataContainer tensor, Unary operation) {
    double[] resultData = Unary.apply(tensor.data, operation);
    return new DataContainer(resultData, tensor.shape);
  }

}
