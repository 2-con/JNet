package com.aufy.jnet.net.functions;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;

/**
 * 
 * ReLU
 */
public class ReLU extends Module {

  @Override
  public Tensor forward(Tensor x) {
    return x.elementwise(
      a -> Math.max(a, 0.0), 
      a -> (a > 0)? 1.0 : 0.0
    );
  }
}
