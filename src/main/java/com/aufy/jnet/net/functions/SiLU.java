package com.aufy.jnet.net.functions;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;

public class SiLU extends Module {

  @Override
  public Tensor forward(Tensor x) {
    return x.elementwise(
      a -> {
        double s = 1.0 / (1.0 + Math.exp(-a));
        return a * s;
      },
      a -> {
        double s = 1.0 / (1.0 + Math.exp(-a));
        return s + a * s * (1.0 - s);
      }
    );
  }
}
