package com.aufy.jnet.tensor.functional.main;

import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.backend.func.Reduction;
import com.aufy.jnet.tensor.core.impl.DataContainer;

public class CoreReductionOps {
  public static DataContainer reduce(DataContainer tensor, Reduction operation, int... axes) {
    return new DataContainer(Reduction.apply(tensor.data, tensor.shape, tensor.strides, axes, operation, true), Shaping.getSurvivors(tensor.shape, axes));
  }

}
