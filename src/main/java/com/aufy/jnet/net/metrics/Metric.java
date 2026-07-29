package com.aufy.jnet.net.metrics;

import com.aufy.jnet.Tensor;

public abstract class Metric {
  public abstract double compute(
    Tensor prediction,
    Tensor target
  );

}
