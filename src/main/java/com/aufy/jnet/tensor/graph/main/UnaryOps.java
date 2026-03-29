package com.aufy.jnet.tensor.graph.main;

import java.util.List;

import com.aufy.jnet.tensor.core.backend.func.Unary;
import com.aufy.jnet.tensor.core.impl.RawTensor;
import com.aufy.jnet.tensor.core.impl.CoreTensor;
import com.aufy.jnet.tensor.functional.main.CoreBinaryOps;
import com.aufy.jnet.tensor.functional.main.CoreUnaryOps;

public class UnaryOps {
  /*
  dont go overboard with the additions, Tensor will do the job. just add the core stuff and Tensor will do the rest
  */
  
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
}
