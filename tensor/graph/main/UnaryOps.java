package tensor.graph.main;

import java.util.List;

import tensor.core.backend.func.Unary;
import tensor.core.impl.DataContainer;
import tensor.core.impl.TensorCore;
import tensor.functional.main.CoreBinaryOps;
import tensor.functional.main.CoreUnaryOps;

public class UnaryOps {
  // ==============================================================================================
  // GENERIC
  // ==============================================================================================

  public static TensorCore apply(TensorCore tensor, Unary operation, Unary derivative) {
    TensorCore out = new TensorCore(CoreUnaryOps.apply(tensor.core, operation));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        DataContainer derivativeCore = CoreUnaryOps.apply(tensor.core, derivative);
        DataContainer gradInputCore = CoreBinaryOps.elementwise(grad.core, derivativeCore, (a, b) -> a * b);
        tensor.accumulate(new TensorCore(gradInputCore));
      };
    }

    return out;
  }

  // ==============================================================================================
  // IMPLEMENTATION
  // ==============================================================================================

  public static TensorCore add(TensorCore tensor, double scalar) {
    return apply(tensor, (a) -> a + scalar, (a) -> 1.0);
  }

  public static TensorCore mul(TensorCore tensor, double scalar) {
    return apply(tensor, (a) -> a * scalar, (a) -> scalar);
  }
  
  public static TensorCore pow(TensorCore tensor, double scalar) {
    return apply(tensor, (a) -> Math.pow(a, scalar), (a) -> scalar * Math.pow(a, scalar - 1.0));
  }
}
