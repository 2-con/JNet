package com.aufy.jnet.tensor.graph.main;

import java.util.List;

import com.aufy.jnet.tensor.core.backend.compute.Engine;
import com.aufy.jnet.tensor.core.backend.func.Reduction;
import com.aufy.jnet.tensor.core.impl.CoreTensor;
import com.aufy.jnet.tensor.functional.main.CoreReductionOps;

public class ReductionOps {
  public static CoreTensor reduce(CoreTensor tensor, Reduction operation, int... axes) {
    return new CoreTensor(CoreReductionOps.reduce(tensor.core, operation, axes)).noGrad();
  }

  public static CoreTensor sum(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(CoreReductionOps.reduce(tensor.core, ReductionOps::sum, axes));
    out.requiresGrad = tensor.requiresGrad;

    if (out.requiresGrad) {
      CoreTensor input = tensor;
      out.parents = List.of(input);

      out.derivative = (grad) -> {
        double[] gradInput = Engine.transformData(grad.dump(), grad.shape, input.shape);
        input.accumulate(new CoreTensor(gradInput, input.shape));
      };
    }

    return out;
  }

  // FIXME: prod is wrong: 1 zero = all grads go here, 2+ zeros = no grad. currently if there are any zeros, NaNs will happen bc 1/0
  public static CoreTensor prod(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(CoreReductionOps.reduce(tensor.core, ReductionOps::prod, axes));

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

        tensor.accumulate(new CoreTensor(gradInput, tensor.shape));
      };
    }

    return out;
  }

  // FIXME:  min and max grads are still wrong: they both assume global reduction when its not guaranteed
  public static CoreTensor max(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(CoreReductionOps.reduce(tensor.core, ReductionOps::max, axes));

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

        tensor.accumulate(new CoreTensor(gradInput, tensor.shape));
      };
    }

    return out;
  }

  public static CoreTensor min(CoreTensor tensor, int... axes) {
    CoreTensor out = new CoreTensor(CoreReductionOps.reduce(tensor.core, ReductionOps::min, axes));

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

        tensor.accumulate(new CoreTensor(gradInput, tensor.shape));
      };
    }

    return out;
  }

  // ==============================================================================================
  // INTERNAL HELPER METHODS
  // ==============================================================================================

  private static double prod(double[] array) {
    double ans = 1;
    for (double n : array) ans *= n;
    return ans;
  }

  private static double sum(double[] array) {
    double ans = 0;
    for (double n : array) ans += n;
    return ans;
  }

  private static double min(double[] array) {
    double ans = Double.MAX_VALUE;
    for (double n : array) ans = (n < ans) ? n : ans;
    return ans;
  }

  private static double max(double[] array) {
    double ans = Double.MIN_VALUE;
    for (double n : array) ans = (n > ans) ? n : ans;
    return ans;
  }

}
