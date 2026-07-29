package com.aufy.jnet.net.functions;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;

public class LeakyReLU extends Module {
  private final double alpha;
  
  public LeakyReLU() {
    this.alpha = 0.1;
  }

  public LeakyReLU(double alpha) {
    this.alpha = alpha;
  }

  @Override
  public Tensor forward(Tensor x) {
    return x.elementwise(
      a -> Math.max(a, alpha), 
      a -> (a > 0)? 1.0 : alpha
    );
  }
}
