package com.aufy.jnet.net.functions;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;

public class Mish extends Module {

  @Override
  public Tensor forward(Tensor x) {
    return x.elementwise(
      a -> a * Math.tanh(Math.log1p(Math.exp(a))),
      a -> {
        double sp = Math.log1p(Math.exp(a));
        double tsp = Math.tanh(sp);
        double sigmoid = 1.0 / (1.0 + Math.exp(-a));
        return tsp + a * sigmoid * (1 - tsp * tsp);
      }
    );

  }
}
