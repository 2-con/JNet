package tensor.graph.init;

import tensor.core.backend.compute.Generator;
import tensor.core.impl.TensorCore;

public class TensorCoreGenerator {
  public static TensorCore randomUniform(int... shape) {return new TensorCore(Generator.generateUniform(shape), shape);}
  public static TensorCore randomNormal(int... shape) {return new TensorCore(Generator.generateGaussian(shape), shape);}
  public static TensorCore randomExponential(int... shape) {return new TensorCore(Generator.generateExponential(shape), shape);}
  public static TensorCore zeros(int... shape) {return new TensorCore(Generator.zeros(shape), shape);}
  public static TensorCore zerosLike(TensorCore tensor) {return zeros(tensor.shape);}
  public static TensorCore ones(int... shape) {return new TensorCore(Generator.ones(shape), shape);}
  public static TensorCore onesLike(TensorCore tensor) {return ones(tensor.shape);}
}
