package com.aufy.jnet.core.tensor.graph.main;

import java.util.List;
import java.util.function.BiFunction;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.scalarops.Logic;
import com.aufy.jnet.core.tensor.core.backend.interfaces.Reduction;
import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;
import com.aufy.jnet.core.tensor.functional.main.RawBinaryOps;
import com.aufy.jnet.core.tensor.functional.main.RawReductionOps;
import com.aufy.jnet.core.tensor.functional.main.RawShapeOps;
import com.aufy.jnet.core.tensor.functional.main.RawUnaryOps;
import com.aufy.jnet.statistics.univariate.Moments;

public class ReductionOps {
  
  public static CoreTensor reduce(CoreTensor tensor, Reduction operation, BiFunction<RawTensor, RawTensor, RawTensor> derivative, int... axes) {
    CoreTensor out = new CoreTensor(RawReductionOps.reduce(tensor.core, operation, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor expandedGrads = grad.core.broadcast(tensor.shape);
        RawTensor expandedOutput = out.core.broadcast(tensor.shape);

        RawTensor localDerivative = derivative.apply(expandedOutput, tensor.core);
        tensor.accumulate(new CoreTensor(localDerivative.hadamard(expandedGrads)));
      };
    }

    return out;

  }

  public static CoreTensor mean(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(RawReductionOps.reduce(tensor.core, Moments::mean, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor expandedGrads = grad.core.broadcast(tensor.shape);
        int size;

        if (axes.length == 0) {
          size = 1;  
        } else {
          size = Reductions.prod(Tools.gather(tensor.shape, axes));
        } 

        tensor.accumulate(new CoreTensor(expandedGrads.hadamard(tensor.core).mul(1.0/size)));
      };
    }

    return out;
  }

  public static CoreTensor sum(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(RawReductionOps.reduce(tensor.core, Reductions::sum, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        tensor.accumulate(ShapeOps.broadcast(grad, tensor.shape));
      };
    }

    return out;
  }

  public static CoreTensor prod(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(RawReductionOps.reduce(tensor.core, Reductions::prod, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor expandedGrads = grad.core.broadcast(tensor.shape);
        RawTensor expandedOutput = out.core.broadcast(tensor.shape);

        RawTensor naiveDerivative = expandedOutput.elementwise(tensor.core, (x, y) -> {
          if (y == 0) {
            return 0.0;
          } else {
            return x / y;
          }
        }).hadamard(expandedGrads);

        // isolate lone zeros in each slice ======================================

        RawTensor onlyOneZeroSlice = RawShapeOps.broadcast(
          RawReductionOps.reduce(tensor.core, data -> (Tools.countContains(data, 0) != 1) ? 0.0 : 1.0, axes), 
          tensor.shape
        );
        RawTensor isItZero = RawUnaryOps.elementwise(tensor.core, x -> (x == 0) ? 1.0 : 0.0);
        RawTensor onlyLoneZeros = onlyOneZeroSlice.hadamard(isItZero);

        // zeroless prod =========================================================

        RawTensor zerolessInputTensor = RawUnaryOps.elementwise(tensor.core, x -> (x == 0) ? 1.0 : x);
        RawTensor zerolessDerivative = zerolessInputTensor.prod(axes).broadcast(tensor.shape);

        // correct derivative for lone zeros =====================================

        RawTensor loneZeroDerivative = onlyLoneZeros.hadamard(zerolessDerivative);

        tensor.accumulate(new CoreTensor(naiveDerivative.add(loneZeroDerivative)));
      };
    }

    return out;
  }

  public static CoreTensor max(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(RawReductionOps.reduce(tensor.core, Reductions::max, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor expandedGrads = grad.core.broadcast(tensor.shape);
        RawTensor expandedOutput = out.core.broadcast(tensor.shape);
        
        RawTensor mask = RawBinaryOps.elementwise(tensor.core, expandedOutput, Logic::equal);
        RawTensor counts = mask.sum(axes).broadcast(mask.getShape());

        tensor.accumulate(new CoreTensor(expandedGrads.div(counts).hadamard(mask)));
      };
    }

    return out;
  }

  public static CoreTensor min(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(RawReductionOps.reduce(tensor.core, Reductions::min, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        RawTensor expandedGrads = grad.core.broadcast(tensor.shape);
        RawTensor expandedOutput = out.core.broadcast(tensor.shape);
        
        RawTensor mask = RawBinaryOps.elementwise(tensor.core, expandedOutput, Logic::equal);
        RawTensor counts = mask.sum(axes).broadcast(mask.getShape());

        tensor.accumulate(new CoreTensor(expandedGrads.div(counts).hadamard(mask)));
      };
    }

    return out;
  }

}
