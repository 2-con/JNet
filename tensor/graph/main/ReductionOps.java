package tensor.graph.main;

import java.util.List;

import stats.Statistics;
import tensor.core.backend.compute.Engine;
import tensor.core.backend.func.Reduction;
import tensor.core.impl.TensorCore;
import tensor.functional.main.CoreReductionOps;

public class ReductionOps {
  public static TensorCore reduce(TensorCore tensor, Reduction operation, int... axes) {
    return new TensorCore(CoreReductionOps.reduce(tensor.core, operation, axes)).noGrad();
  }

  public static TensorCore sum(TensorCore tensor, int... axes) {
    TensorCore out = new TensorCore(CoreReductionOps.reduce(tensor.core, Statistics::sum, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      TensorCore input = tensor;
      out.parents = List.of(input);

      out.derivative = (grad) -> {
        double[] gradInput = Engine.transformData(grad.dump(), grad.shape, input.shape);
        input.accumulate(new TensorCore(gradInput, input.shape));
      };
    }

    return out;
  }

  // FIXME: prod is wrong: 1 zero = all grads go here, 2+ zeros = no grad. currently if there are any zeros, NaNs will happen bc 1/0
  public static TensorCore prod(TensorCore tensor, int... axes) {
    TensorCore out = new TensorCore(CoreReductionOps.reduce(tensor.core, Statistics::prod, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        double[] gradExpanded = Engine.transformData(grad.dump(), grad.shape, tensor.shape);
        double[] yExpanded = Engine.transformData(out.core.dump(), out.shape, tensor.shape);

        double[] xData = tensor.dump();
        double[] gradInput = new double[tensor.size];

        for (int i = 0; i < gradInput.length; i++) {
          gradInput[i] = gradExpanded[i] * (yExpanded[i] / xData[i]);
        }

        tensor.accumulate(new TensorCore(gradInput, tensor.shape));
      };
    }

    return out;
  }

  // FIXME:  min and max grads are still wrong: they both assume global reduction when its not guaranteed
  public static TensorCore max(TensorCore tensor, int... axes) {
    TensorCore out = new TensorCore(CoreReductionOps.reduce(tensor.core, Statistics::max, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        double[] gradExpanded = Engine.transformData(grad.dump(), grad.shape, tensor.shape);
        double[] yExpanded = Engine.transformData(out.core.dump(), out.shape, tensor.shape);

        double[] xData = tensor.dump();
        double[] gradInput = new double[tensor.size];

        int count = 0;
        for (double n: xData) count += (n == yExpanded[0]) ? 1 : 0;
        for (int i = 0; i < gradInput.length; i++) gradInput[i] = (xData[i] == yExpanded[i]) ? gradExpanded[i] / count : 0;

        tensor.accumulate(new TensorCore(gradInput, tensor.shape));
      };
    }

    return out;
  }

  public static TensorCore min(TensorCore tensor, int... axes) {
    TensorCore out = new TensorCore(CoreReductionOps.reduce(tensor.core, Statistics::min, axes));

    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      out.parents = List.of(tensor);

      out.derivative = (grad) -> {
        double[] gradExpanded = Engine.transformData(grad.dump(), grad.shape, tensor.shape);
        double[] yExpanded = Engine.transformData(out.core.dump(), out.shape, tensor.shape);

        double[] xData = tensor.dump();
        double[] gradInput = new double[tensor.size];

        int count = 0;
        for (double n: xData) count += (n == yExpanded[0]) ? 1 : 0;
        for (int i = 0; i < gradInput.length; i++) gradInput[i] = (xData[i] == yExpanded[i]) ? gradExpanded[i] / count : 0;

        tensor.accumulate(new TensorCore(gradInput, tensor.shape));
      };
    }

    return out;
  }

}
