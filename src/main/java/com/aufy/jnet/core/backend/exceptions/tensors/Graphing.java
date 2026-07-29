package com.aufy.jnet.core.backend.exceptions.tensors;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Utility class for validating tensor graph and automatic differentiation states.
 */
public class Graphing {
  /*
   * still dont know wether these should be defaults or should be implemented
   * inside the Tensor implementation itself. imports
   * would be hell if its automatic yet stuff should be here to keep consistency
   */

  /**
   * Validates that a tensor tracks gradients and supports differentiation.
   *
   * @param operationName the name of the calling operation
   * @param requiresGrad true if the tensor supports gradients
   * @throws IllegalStateException if the tensor is not differentiable
   */
  public static void verifyDifferentiable(String operationName, boolean requiresGrad) throws IllegalStateException {
    if (!requiresGrad) {
      throw new IllegalStateException(Message.crash("tensor graphing", "Illegal operation", operationName, "This tensor does not require gradients"));
    }
  }

  /**
   * Validates that a tensor has the history required for a backward pass.
   *
   * @param operationName the name of the calling operation
   * @param parents the parent nodes in the computation graph
   * @param derivative the backward derivative function or data
   * @throws IllegalStateException if both arguments are null
   */
  public static void verifyBackwardsInheritance(String operationName, Object parents, Object derivative) throws IllegalStateException {
    if (parents == null && derivative == null) {
      throw new IllegalStateException(Message.crash("tensor graphing", "Illegal operation", operationName, "This tensor is the result of a non-differentiable operation"));
    }
  }

}

