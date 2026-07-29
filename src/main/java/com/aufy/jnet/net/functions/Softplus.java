package com.aufy.jnet.net.functions;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;

public class Softplus extends Module {

  @Override
  public Tensor forward(Tensor x) {
    return x.elementwise(
      a -> Math.log(Math.exp(a)),
      a -> 1.0 / (1.0 + Math.exp(-a))
    );
  }
}
