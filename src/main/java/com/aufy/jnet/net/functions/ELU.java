package com.aufy.jnet.net.functions;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;

public class ELU extends Module {
  private final double alpha;

  public ELU() {
    this.alpha = 1.0;
  }

  public ELU(double alpha) {
    this.alpha = alpha;
  }

  @Override
  public Tensor forward(Tensor x) {
    return x.elementwise(
      a -> a > 0 ? a   : alpha * (Math.exp(a) - 1),
      a -> a > 0 ? 1.0 : alpha * Math.exp(a)
    );
  }
}
