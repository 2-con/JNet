package com.aufy.jnet.tensor.graph.main;

import java.util.List;

import com.aufy.jnet.tensor.core.backend.func.Unary;
import com.aufy.jnet.tensor.core.impl.RawTensor;
import com.aufy.jnet.tensor.core.impl.CoreTensor;
import com.aufy.jnet.tensor.functional.main.CoreBinaryOps;
import com.aufy.jnet.tensor.functional.main.CoreUnaryOps;

public class UnaryOps {
  // ==============================================================================================
  // GENERIC
  // ==============================================================================================

  public static CoreTensor apply(CoreTensor tensor, Unary operation, Unary derivative) {
    CoreTensor out = new CoreTensor(CoreUnaryOps.apply(tensor.core, operation));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor derivativeCore = CoreUnaryOps.apply(tensor.core, derivative);
        RawTensor gradInputCore = CoreBinaryOps.elementwise(grad.core, derivativeCore, (a, b) -> a * b);
        tensor.accumulate(new CoreTensor(gradInputCore));
      };
    }

    return out;
  }

  // ==============================================================================================
  // IMPLEMENTATION
  // ==============================================================================================

  public static CoreTensor add(CoreTensor tensor, double scalar) {
    return apply(tensor, (a) -> a + scalar, (a) -> 1.0);
  }

  public static CoreTensor mul(CoreTensor tensor, double scalar) {
    return apply(tensor, (a) -> a * scalar, (a) -> scalar);
  }
  
  public static CoreTensor pow(CoreTensor tensor, double scalar) {
    return apply(tensor, (a) -> Math.pow(a, scalar), (a) -> scalar * Math.pow(a, scalar - 1.0));
  }
}
