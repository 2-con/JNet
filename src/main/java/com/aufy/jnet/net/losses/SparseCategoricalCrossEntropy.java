package com.aufy.jnet.net.losses;

import com.aufy.jnet.Tensor;

public class SparseCategoricalCrossEntropy extends Loss{
  @Override
  public Tensor compute(Tensor prediction, Tensor target) {
    System.out.println("SCCE loss is a work in progress");
    return prediction.sub(target).abs().sum(0);
  }
}
