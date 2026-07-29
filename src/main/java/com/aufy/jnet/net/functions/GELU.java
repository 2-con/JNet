package com.aufy.jnet.net.functions;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.net.Module;

public class GELU extends Module {

  @Override
  public Tensor forward(Tensor x) { // just an approximation
    return x.elementwise(
      a -> {
        double c = Math.sqrt(2.0 / Math.PI);
        return 0.5 * a * (1.0 + Math.tanh(c * (a + 0.044715 * a * a * a)));
      },
      a -> {
        double c = Math.sqrt(2.0 / Math.PI);

        double a3 = a * a * a;
        double inner = c * (a + 0.044715 * a3);

        double t = Math.tanh(inner);

        double sech2 = 1.0 - t * t;

        double innerPrime = c * (1.0 + 3.0 * 0.044715 * a * a);

        return 0.5 * (1 + t) + 0.5 * a * sech2 * innerPrime;
      }
    );
  }
}
