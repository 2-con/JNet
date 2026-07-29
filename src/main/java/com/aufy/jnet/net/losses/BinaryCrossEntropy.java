package com.aufy.jnet.net.losses;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.core.backend.scalarops.Unary;

public class BinaryCrossEntropy extends Loss{
  @Override
  public Tensor compute(Tensor prediction, Tensor target) {
    return Tensor.apply(prediction, target,
      (p, t) -> {
        var probability = p.elementwise(Unary::sigmoid).clip(0.0001, 0.9999);
        var loss1 = probability.ln().hadamard(t);
        var loss2 = p.onesLike().sub(probability).ln().hadamard(t.onesLike().sub(t));
        var loss = loss1.add(loss2);

        return loss.sum(0,1).mul(1.0/p.getShape()[0]);
      },

      (p, t) -> {
        var sigmoided = p.elementwise(Unary::sigmoid).clip(0.0001, 0.9999);
        return sigmoided.sub(t).mul(1.0/p.getShape()[0]);
      },
      (p, t) -> p.zerosLike() // target wont need gradients
    );
  }
}
