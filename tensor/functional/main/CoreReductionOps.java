package tensor.functional.main;

import tensor.core.backend.compute.Shaping;
import tensor.core.backend.func.Reduction;
import tensor.core.impl.DataContainer;

public class CoreReductionOps {
  public static DataContainer reduce(DataContainer tensor, Reduction operation, int... axes) {
    return new DataContainer(Reduction.apply(tensor.data, tensor.shape, tensor.strides, axes, operation, true), Shaping.getSurvivors(tensor.shape, axes));
  }

}
