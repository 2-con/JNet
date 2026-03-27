package com.aufy.jnet.tensor.core.exception;

public class Graph {
  public static void differentiable(boolean requiresGrad) {
    if (!requiresGrad) {
      throw new IllegalStateException(".backward() cannot be called on a tensor that does not require gradients");
    }
  }
  
  public static void inheritanceCheck(Object parents, Object derivative) {
    if (parents == null && derivative == null) {
      throw new IllegalStateException("This tensor is the result of a non-differentiable operation");
    }
  }
}
