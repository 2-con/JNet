package com.aufy.jnet.net.functions;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;

public class Softsign extends Module {

  @Override
  public Tensor forward(Tensor x) {
    return x.elementwise(
      a -> a / (Math.abs(a) + 1),
      a -> 1.0 / ((Math.abs(a) + 1) * (Math.abs(a) + 1))
    );
  }
}
