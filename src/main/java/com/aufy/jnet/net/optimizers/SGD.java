package com.aufy.jnet.net.optimizers;

import java.util.List;

import com.aufy.jnet.net.Parameter;

public class SGD extends Optimizer {

  public SGD(List<Parameter> params, double learningRate) {
    super(params, learningRate);
  }

  @Override
  public void step() {
    for (Parameter p : parameters) {
      p.core = p.core.sub(p.core.grad().mul(this.learningRate));
    }
  }
}
