package com.aufy.jnet.net.functions;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;

public class Tanh extends Module {

  @Override
  public Tensor forward(Tensor x) {
    return x.elementwise(
      Math::tanh,
      a -> 1.0 - (Math.tanh(a) * Math.tanh(a))
    );
  }
}
