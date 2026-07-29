package com.aufy.jnet.net.optimizers;

import java.util.List;

import com.aufy.jnet.net.Parameter;

public abstract class Optimizer {
  /* 
  literally pytorch. the JXNet way of doing it is too specific and inflexible
   */
  
  protected final List<Parameter> parameters;

  protected double learningRate;

  public Optimizer(List<Parameter> parameters, double learningRate) {
    this.parameters = parameters;
    this.learningRate = learningRate;
  }

  public abstract void step();

  public void zeroGrad() {
    for (Parameter p : parameters) p.core.zeroGrad();
  }

}
