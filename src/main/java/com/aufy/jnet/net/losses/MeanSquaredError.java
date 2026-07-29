package com.aufy.jnet.net.losses;

import com.aufy.jnet.Tensor;

public class MeanSquaredError extends Loss{
  @Override
  public Tensor compute(Tensor prediction, Tensor target) {
    int size = prediction.size;
    // return prediction.sub(target).pow(2).sum(0,1).mul(1.0/size);

    return Tensor.apply(prediction, target,
      (p, t) -> {
        return p.sub(t).pow(2).sum(0,1).mul(1.0/size);
      },

      (p, t) -> p.sub(t).mul(2.0/size),
      (p, t) -> p.zerosLike() // target wont need gradients
    );
  }
}
