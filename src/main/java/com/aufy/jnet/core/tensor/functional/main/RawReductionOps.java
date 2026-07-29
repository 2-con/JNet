package com.aufy.jnet.core.tensor.functional.main;

import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.core.tensor.core.backend.interfaces.Reduction;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;

public class RawReductionOps {
  /*
  keep this simple because coretensor will implement the rest
   */

  public static RawTensor reduce(RawTensor tensor, Reduction operation, int... axes) {
    return new RawTensor(Reduction.apply(tensor.dump(), tensor.getShape(), tensor.getStrides(), axes, operation, true), Shaping.reduceAxes(tensor.getShape(), axes));
  }

}
