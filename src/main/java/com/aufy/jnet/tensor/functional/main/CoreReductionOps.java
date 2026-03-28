package com.aufy.jnet.tensor.functional.main;

import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.backend.func.Reduction;
import com.aufy.jnet.tensor.core.impl.RawTensor;

public class CoreReductionOps {
  /*
  keep this simple because coretensor will implement the rest
   */

  public static RawTensor reduce(RawTensor tensor, Reduction operation, int... axes) {
    return new RawTensor(Reduction.apply(tensor.data, tensor.shape, tensor.strides, axes, operation, true), Shaping.getSurvivors(tensor.shape, axes));
  }

}
