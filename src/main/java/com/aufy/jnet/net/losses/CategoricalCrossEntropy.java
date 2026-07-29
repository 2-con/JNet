package com.aufy.jnet.net.losses;

import com.aufy.jnet.Tensor;

public class CategoricalCrossEntropy extends Loss{
  @Override
  public Tensor compute(Tensor prediction, Tensor target) {
    return Tensor.apply(prediction, target,
      (p, t) -> {
        var softmax = p.exp().div(p.exp().sum(1)).clip(0.0001, 0.9999);
        var loss = softmax.ln().mul(-1.0).hadamard(t);
        var averaged = loss.sum(0,1).mul(1.0/p.getShape()[0]);

        return softmax;
      },

      (p, t) -> {
        var softmax = p.exp().div(p.exp().sum(1)).clip(0.0001, 0.9999);
        return softmax.sub(t).mul(1.0/p.getShape()[0]);
      },
      (p, t) -> p.zerosLike() // target wont need gradients
    );
  }
}
