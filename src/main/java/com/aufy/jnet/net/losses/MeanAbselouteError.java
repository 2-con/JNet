package com.aufy.jnet.net.losses;

import com.aufy.jnet.Tensor;

public class MeanAbselouteError extends Loss{
  @Override
  public Tensor compute(Tensor prediction, Tensor target) {
    int size = prediction.size;
    return prediction.sub(target).abs().sum(0).mul(1.0/size);
  }
}
