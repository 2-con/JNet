package com.aufy.jnet.stats.tensor;

import com.aufy.jnet.stats.primitive.Statistics;
import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.impl.CoreTensor;
import com.aufy.jnet.tensor.functional.main.CoreReductionOps;
import com.aufy.jnet.tensor.graph.main.BinaryOps;
import com.aufy.jnet.tensor.graph.main.ReductionOps;
import com.aufy.jnet.tensor.graph.main.UnaryOps;

public class Measure {
  public static CoreTensor mean(CoreTensor tensor, int... axes) {
    CoreTensor sum = ReductionOps.sum(tensor, axes);
    int reductionSize = Statistics.prod(Shaping.getSubShape(tensor.shape, axes));

    return UnaryOps.mul(sum, 1.0 / reductionSize);
  }

  public static CoreTensor variance(CoreTensor tensor, int... axes) {
    CoreTensor mean = mean(tensor, axes);

    CoreTensor centered = BinaryOps.sub(tensor, mean);
    CoreTensor sq = UnaryOps.pow(centered, 2);

    return mean(sq, axes);
  }

  public static CoreTensor stdev(CoreTensor tensor, int... axes) {
    return UnaryOps.pow(variance(tensor, axes), 0.5);
  }
 
  public static CoreTensor skew(CoreTensor tensor, int... axes) {
    CoreTensor mean = mean(tensor, axes);
    CoreTensor centered = BinaryOps.sub(tensor, mean);

    CoreTensor m3 = mean(UnaryOps.pow(centered, 3), axes);
    CoreTensor stdev = stdev(tensor, axes);

    return BinaryOps.div(m3, UnaryOps.pow(stdev, 3));
  }

  public static CoreTensor kurtosis(CoreTensor tensor, int... axes) {
    CoreTensor mean = mean(tensor, axes);
    CoreTensor centered = BinaryOps.sub(tensor, mean);

    CoreTensor m4 = mean(UnaryOps.pow(centered, 4), axes);
    CoreTensor stdev = stdev(tensor, axes);

    return BinaryOps.div(m4, UnaryOps.pow(stdev, 4));
  }

  // non-differentiable

  public static CoreTensor range(CoreTensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("range() is not differentiable");
    }

    return new CoreTensor(CoreReductionOps.reduce(tensor.core, Statistics::range, axes)).noGrad();
  }

  public static CoreTensor median(CoreTensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("median() is not differentiable");
    }

    return new CoreTensor(CoreReductionOps.reduce(tensor.core, Statistics::median, axes)).noGrad();
  }

  public static CoreTensor mode(CoreTensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("mode() is not differentiable");
    }

    return new CoreTensor(CoreReductionOps.reduce(tensor.core, Statistics::mode, axes)).noGrad();
  }

  public static CoreTensor quartile1(CoreTensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("quartile1() is not differentiable");
    }

    return new CoreTensor(CoreReductionOps.reduce(tensor.core, Statistics::quartile1, axes)).noGrad();
  }

  public static CoreTensor quartile3(CoreTensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("quartile3() is not differentiable");
    }

    return new CoreTensor(CoreReductionOps.reduce(tensor.core, Statistics::quartile3, axes)).noGrad();
  }

  public static CoreTensor iqr(CoreTensor tensor, int... axes) {
    if (tensor.requiresGrad) {
      throw new UnsupportedOperationException("iqr() is not differentiable");
    }

    return new CoreTensor(CoreReductionOps.reduce(tensor.core, Statistics::iqr, axes)).noGrad();
  }
}
