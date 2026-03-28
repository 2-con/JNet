package com.aufy.jnet.tensor.tools;

import java.util.Arrays;

import com.aufy.jnet.tensor.core.impl.CoreTensor;
import com.aufy.jnet.tensor.core.impl.RawTensor;

public class Debug {
  /*
  add the basic checkers for now
   */

  public static void parse(RawTensor tensor) {
    double[] data = tensor.dump();
    for (double d : data) {
      if (Double.isNaN(d)) throw new RuntimeException("Detected NaN");
      if (Double.isInfinite(d)) System.err.println("Warning: Infinity detected");
    }
  }

  public static void parse(CoreTensor tensor) {
    double[] data = tensor.dump();
    for (double d : data) {
      if (Double.isNaN(d)) throw new RuntimeException("Detected NaN");
      if (Double.isInfinite(d)) System.err.println("Warning: Infinity detected");
    }
  }

  public static double sparsity(RawTensor tensor) {
    double[] data = tensor.dump();
    return (double) Arrays.stream(data).filter(d -> d == 0).count() / data.length;
  }

  public static double sparsity(CoreTensor tensor) {
    double[] data = tensor.dump();
    return (double) Arrays.stream(data).filter(d -> d == 0).count() / data.length;
  }
}
