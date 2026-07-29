package com.aufy.jnet.core.tensor.functional.init;

import com.aufy.jnet.core.tensor.core.backend.compute.Generator;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;

public class RawTensorGenerator {
  /*
  possibly the most useless bundle of methods ever made: core tensor already has these? ill find an implementation that 
  makes sense soon
   */

  public static RawTensor zeros(int... shape) {return new RawTensor(Generator.zeros(shape), shape);}
  public static RawTensor zerosLike(RawTensor tensor) {return zeros(tensor.getShape());}
  public static RawTensor ones(int... shape) {return new RawTensor(Generator.ones(shape), shape);}
  public static RawTensor onesLike(RawTensor tensor) {return ones(tensor.getShape());}
}
