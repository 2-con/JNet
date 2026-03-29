package com.aufy.jnet.tensor.core.exception;

public class Calculus {
  /*
  still dont know wether these should be defaults or should be implemented inside the Tensor implementation itself. imports
  would be hell if its automatic yet stuff should be here to keep consistency
   */

  private static String namingMessage(String operationName) {
    return (operationName == null || operationName.isBlank()) ? "" : " for " + operationName;
  }
  
  public static void verifyDifferentiable(String operationName, boolean requiresGrad) throws IllegalStateException {
    if (!requiresGrad) {
      throw new IllegalStateException("Illegal operation" + namingMessage(operationName) + ": a tensor that does not require gradients cannot be differentiated");
    }
  }
  
  public static void verifyBackwardsInheritance(String operationName, Object parents, Object derivative) throws IllegalStateException {
    if (parents == null && derivative == null) {
      throw new IllegalStateException("Illegal operation" + namingMessage(operationName) + ": This tensor is the result of a non-differentiable operation");
    }
  }
}
