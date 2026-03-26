package com.aufy.jnet.tensor.functional.init;

import com.aufy.jnet.tensor.core.backend.compute.Generator;
import com.aufy.jnet.tensor.core.impl.DataContainer;

public class DataContainerGenerator {
  public static DataContainer randomUniform(int... shape) {return new DataContainer(Generator.generateUniform(shape), shape);}
  public static DataContainer randomNormal(int... shape) {return new DataContainer(Generator.generateGaussian(shape), shape);}
  public static DataContainer randomExponential(int... shape) {return new DataContainer(Generator.generateExponential(shape), shape);}
  public static DataContainer zeros(int... shape) {return new DataContainer(Generator.zeros(shape), shape);}
  public static DataContainer zerosLike(DataContainer tensor) {return zeros(tensor.shape);}
  public static DataContainer ones(int... shape) {return new DataContainer(Generator.ones(shape), shape);}
  public static DataContainer onesLike(DataContainer tensor) {return ones(tensor.shape);}
}
