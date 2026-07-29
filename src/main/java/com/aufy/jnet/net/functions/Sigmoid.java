package com.aufy.jnet.net.functions;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;

public class Sigmoid extends Module {

  @Override
  public Tensor forward(Tensor x) {
    return x.elementwise(
      a -> 1.0 / (1.0 + Math.exp(-a)),
      a -> { // derivative: sigmoid * (1 - sigmoid)
        double s = 1.0 / (1.0 + Math.exp(-a)); 
        return s * (1.0 - s);
      }
    );
  }
}
