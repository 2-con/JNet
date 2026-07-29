package com.aufy.jnet.net.functions;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;

public class SELU extends Module {
  private final double lambda;
  private final double alpha;

  public SELU() {
    this.lambda = 1.0507009873554804934193349852946;
    this.alpha = 1.6732632423543772848170429916717;
  }

  public SELU(double lambda, double alpha) {
    this.lambda = lambda;
    this.alpha = alpha;
  }

  @Override
  public Tensor forward(Tensor x) {
    return x.elementwise(
      a -> (a > 0)? lambda * a : lambda * alpha * (Math.exp(a) - 1.0), 
      a -> (a > 0)? lambda : lambda * alpha * Math.exp(a)
    );
  }
}
