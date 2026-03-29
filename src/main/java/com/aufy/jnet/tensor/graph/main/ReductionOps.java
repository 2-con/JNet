package com.aufy.jnet.tensor.graph.main;

import java.util.Arrays;
import java.util.List;

import com.aufy.jnet.tensor.core.backend.compute.Engine;
import com.aufy.jnet.tensor.core.backend.func.Reduction;
import com.aufy.jnet.tensor.core.backend.util.ArrayOps;
import com.aufy.jnet.tensor.core.backend.util.ArrayTools;
import com.aufy.jnet.tensor.core.impl.CoreTensor;
import com.aufy.jnet.tensor.core.impl.RawTensor;
import com.aufy.jnet.tensor.functional.main.CoreReductionOps;

public class ReductionOps {
  /*
  dont go overboard with the additions, Tensor will do the job. just add the core stuff and Tensor will do the rest.

  reductionOps must make implementations because derivatives are not generalzable beyond the basics or composites.
  */
  
  public static CoreTensor reduce(CoreTensor tensor, Reduction operation, int... axes) {
    return new CoreTensor(CoreReductionOps.reduce(tensor.core, operation, axes)).noGrad();
  }

  // ==============================================================================================
  // IMPLEMENTATION
  // ==============================================================================================

  public static CoreTensor sum(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(CoreReductionOps.reduce(tensor.core, ArrayOps::sum, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      CoreTensor input = tensor;
      out.parents = List.of(input);

      out.derivative = (grad) -> {
        double[] gradInput = Engine.broadcast(grad.dump(), grad.shape, input.shape);
        input.accumulate(new CoreTensor(gradInput, input.shape));
      };
    }

    return out;
  }

  public static CoreTensor prod(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(CoreReductionOps.reduce(tensor.core, ArrayOps::prod, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        double[] expandedGrads = Engine.broadcast(grad.dump(), grad.shape, tensor.shape);
        double[] expandedOutput = Engine.broadcast(out.core.dump(), out.shape, tensor.shape);

        double[] xData = tensor.dump();
        double[] gradInput = new double[tensor.size];

        // if there are more than 1 0s, there is no gradient
        if (ArrayTools.countContains(xData, 0) > 1) {
          Arrays.fill(gradInput, 0.0);

        } else if (ArrayTools.countContains(xData, 0) == 1) {
          double prodNonZeros = 1;

          for (int i = 0; i < xData.length; i++) {
            if (xData[i] != 0) {
              prodNonZeros *= xData[i];
            }
          }

          int zeroIndex = ArrayTools.indexOf(xData, 0);
          Arrays.fill(gradInput, 0.0);

          gradInput[zeroIndex] = expandedGrads[zeroIndex] * prodNonZeros;
          
        } else {
          for (int i = 0; i < gradInput.length; i++) {
            gradInput[i] = expandedGrads[i] * (expandedOutput[i] / xData[i]);
          }
        }

        tensor.accumulate(new CoreTensor(gradInput, tensor.shape));
      };
    }

    return out;
  }

  public static CoreTensor max(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(CoreReductionOps.reduce(tensor.core, ArrayOps::max, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        double[] expandedGrads = Engine.broadcast(grad.dump(), grad.shape, tensor.shape);
        double[] expandedOutput = Engine.broadcast(out.core.dump(), out.shape, tensor.shape);

        double[] xData = tensor.dump();
        double[] gradInput = new double[tensor.size];

        double[] mask = new double[tensor.size];
        for (int i = 0; i < xData.length; i++) {
          mask[i] = (xData[i] == expandedOutput[i]) ? 1.0 : 0.0; // check if its max, if so, mask 1, else 0
        }

        RawTensor countsRaw = CoreReductionOps.reduce(new RawTensor(mask, tensor.shape), ArrayOps::sum, axes);
        double[] expandedCounts = Engine.broadcast(countsRaw.dump(), countsRaw.getShape(), tensor.shape);

        // distribute gradients
        for (int i = 0; i < gradInput.length; i++) {
          if (mask[i] > 0) {
            gradInput[i] = expandedGrads[i] / expandedCounts[i];
          } else {
            gradInput[i] = 0;
          }
        }

        tensor.accumulate(new CoreTensor(gradInput, tensor.shape));
      };
    }

    return out;
  }

  public static CoreTensor min(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(CoreReductionOps.reduce(tensor.core, ArrayOps::min, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        double[] expandedGrads = Engine.broadcast(grad.dump(), grad.shape, tensor.shape);
        double[] expandedOutput = Engine.broadcast(out.core.dump(), out.shape, tensor.shape);

        double[] xData = tensor.dump();
        double[] gradInput = new double[tensor.size];

        double[] mask = new double[tensor.size];
        for (int i = 0; i < xData.length; i++) {
          mask[i] = (xData[i] == expandedOutput[i]) ? 1.0 : 0.0; // check if min, if so, mask 1, else 0
        }

        RawTensor countsRaw = CoreReductionOps.reduce(new RawTensor(mask, tensor.shape), ArrayOps::sum, axes);
        double[] expandedCounts = Engine.broadcast(countsRaw.dump(), countsRaw.getShape(), tensor.shape);

        // distribute gradients
        for (int i = 0; i < gradInput.length; i++) {
          if (mask[i] > 0) {
            gradInput[i] = expandedGrads[i] / expandedCounts[i];
          } else {
            gradInput[i] = 0;
          }
        }
        tensor.accumulate(new CoreTensor(gradInput, tensor.shape));
      };
    }

    return out;
  }
}
