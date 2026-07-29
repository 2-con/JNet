package com.aufy.jnet.core.tensor.graph.main;

import java.util.List;
import java.util.function.Function;

import com.aufy.jnet.core.backend.interfaces.UnaryOp;
import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;
import com.aufy.jnet.core.tensor.functional.main.RawBinaryOps;
import com.aufy.jnet.core.tensor.functional.main.RawUnaryOps;

public class UnaryOps {
  /*
  dont go overboard with the additions, Tensor will do the job. just add the core stuff and Tensor will do the rest
  */
  
  public static CoreTensor apply(CoreTensor tensor, Function<RawTensor, RawTensor> operation, Function<RawTensor, RawTensor> derivative) {
    CoreTensor out = new CoreTensor(operation.apply(tensor.core));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor derivativeCore = derivative.apply(grad.core);
        RawTensor gradInputCore = RawBinaryOps.elementwise(grad.core, derivativeCore, (a, b) -> a * b);
        tensor.accumulate(new CoreTensor(gradInputCore));
      };
    }

    return out;
  }

  public static CoreTensor elementwise(CoreTensor tensor, UnaryOp operation, UnaryOp derivative) {
    CoreTensor out = new CoreTensor(RawUnaryOps.elementwise(tensor.core, operation));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor derivativeCore = RawUnaryOps.elementwise(tensor.core, derivative); // dy/dx
        RawTensor gradInputCore = RawBinaryOps.elementwise(grad.core, derivativeCore, (a, b) -> a * b); // grad * dy/dx = grad/dx
        tensor.accumulate(new CoreTensor(gradInputCore));
      };
    }

    return out;
  }
}
