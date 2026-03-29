package com.aufy.jnet.stats.tensor;

import com.aufy.jnet.Tensor;
import com.aufy.jnet.stats.primitive.Statistics;
import com.aufy.jnet.tensor.core.backend.compute.Shaping;

public class Measure {
  public static Tensor mean(Tensor tensor, int... axes) {
    Tensor sum = Tensor.sum(tensor, axes);
    int reductionSize = Statistics.prod(Shaping.getSubShape(tensor.shape, axes));

    return Tensor.mul(sum, 1.0 / reductionSize);
  }

  public static Tensor variance(Tensor tensor, int... axes) {
    Tensor mean = mean(tensor, axes);

    Tensor centered = Tensor.sub(tensor, mean);
    Tensor sq = Tensor.pow(centered, 2);

    return mean(sq, axes);
  }

  public static Tensor stdev(Tensor tensor, int... axes) {
    return Tensor.pow(variance(tensor, axes), 0.5);
  }
 
  public static Tensor skew(Tensor tensor, int... axes) {
    Tensor mean = mean(tensor, axes);
    Tensor centered = Tensor.sub(tensor, mean);

    Tensor m3 = mean(Tensor.pow(centered, 3), axes);
    Tensor stdev = stdev(tensor, axes);

    return Tensor.div(m3, Tensor.pow(stdev, 3));
  }

  public static Tensor kurtosis(Tensor tensor, int... axes) {
    Tensor mean = mean(tensor, axes);
    Tensor centered = Tensor.sub(tensor, mean);

    Tensor m4 = mean(Tensor.pow(centered, 4), axes);
    Tensor stdev = stdev(tensor, axes);

    return Tensor.div(m4, Tensor.pow(stdev, 4));
  }

  // non-differentiable

  public static Tensor range(Tensor tensor, int... axes) {
    if (tensor.enableGrad) {
      throw new UnsupportedOperationException("range() is not differentiable");
    }

    return Tensor.reduce(tensor, Statistics::range, axes);
  }

  public static Tensor median(Tensor tensor, int... axes) {
    if (tensor.enableGrad) {
      throw new UnsupportedOperationException("median() is not differentiable");
    }

    return Tensor.reduce(tensor, Statistics::median, axes);
  }

  public static Tensor mode(Tensor tensor, int... axes) {
    if (tensor.enableGrad) {
      throw new UnsupportedOperationException("mode() is not differentiable");
    }

    return Tensor.reduce(tensor, Statistics::mode, axes);
  }

  public static Tensor quartile1(Tensor tensor, int... axes) {
    if (tensor.enableGrad) {
      throw new UnsupportedOperationException("quartile1() is not differentiable");
    }

    return Tensor.reduce(tensor, Statistics::quartile1, axes);
  }

  public static Tensor quartile3(Tensor tensor, int... axes) {
    if (tensor.enableGrad) {
      throw new UnsupportedOperationException("quartile3() is not differentiable");
    }

    return Tensor.reduce(tensor, Statistics::quartile3, axes);
  }

  public static Tensor iqr(Tensor tensor, int... axes) {
    if (tensor.enableGrad) {
      throw new UnsupportedOperationException("iqr() is not differentiable");
    }

    return Tensor.reduce(tensor, Statistics::iqr, axes);
  }
}
