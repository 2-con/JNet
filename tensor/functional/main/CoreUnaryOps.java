package tensor.functional.main;

import tensor.core.backend.func.Unary;
import tensor.core.impl.DataContainer;

public class CoreUnaryOps {
  public static DataContainer apply(DataContainer tensor, Unary operation) {
    double[] resultData = Unary.apply(tensor.data, operation);
    return new DataContainer(resultData, tensor.shape);
  }

}
