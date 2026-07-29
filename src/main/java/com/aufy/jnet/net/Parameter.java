package com.aufy.jnet.net;

import com.aufy.jnet.Tensor;

/**
 * Represents a trainable parameter. Unlike tensors, this is mutable but is really just a wrapper around a tensor.
 */
public class Parameter {
  public Tensor core;

  public Parameter(Tensor value) {
    this.core = value.requiresGrad();
  }
}
