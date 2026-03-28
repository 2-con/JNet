package com.aufy.jnet.tensor.core.exception;

public class Graph {
  /*
  still dont know wether these should be defaults or should be implemented inside the Tensor implementation itself. imports
  would be hell if its automatic yet stuff should be here to keep consistency
   */
  
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
