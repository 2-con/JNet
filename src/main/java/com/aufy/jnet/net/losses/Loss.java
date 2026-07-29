package com.aufy.jnet.net.losses;

import com.aufy.jnet.Tensor;

public abstract class Loss {
  public abstract Tensor compute(
    Tensor prediction,
    Tensor target
  );
}
