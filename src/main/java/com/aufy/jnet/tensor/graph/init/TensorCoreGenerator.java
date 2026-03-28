package com.aufy.jnet.tensor.graph.init;

import com.aufy.jnet.tensor.core.backend.compute.Generator;
import com.aufy.jnet.tensor.core.impl.CoreTensor;

public class TensorCoreGenerator {
  public static CoreTensor zeros(int... shape) {return new CoreTensor(Generator.zeros(shape), shape);}
  public static CoreTensor zerosLike(CoreTensor tensor) {return zeros(tensor.shape);}
  public static CoreTensor ones(int... shape) {return new CoreTensor(Generator.ones(shape), shape);}
  public static CoreTensor onesLike(CoreTensor tensor) {return ones(tensor.shape);}
}
