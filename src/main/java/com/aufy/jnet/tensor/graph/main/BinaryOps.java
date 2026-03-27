package com.aufy.jnet.tensor.graph.main;

import java.util.List;

import com.aufy.jnet.tensor.core.backend.compute.Engine;
import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.backend.func.Binary;
import com.aufy.jnet.tensor.core.impl.CoreTensor;
import com.aufy.jnet.tensor.core.impl.RawTensor;
import com.aufy.jnet.tensor.functional.main.CoreBinaryOps;
import com.aufy.jnet.tensor.functional.main.CoreShapeOps;

public class BinaryOps {
  // ==============================================================================================
  // GENERIC
  // ==============================================================================================
  
  public static CoreTensor elementwise(CoreTensor tensorA, CoreTensor tensorB, Binary op, Binary dA, Binary dB) {
    int[] broadcastShapeTarget = Shaping.broadcastedShape(tensorA.shape, tensorB.shape);
    
    RawTensor coreA = CoreShapeOps.broadcast(tensorA.core, broadcastShapeTarget);
    RawTensor coreB = CoreShapeOps.broadcast(tensorB.core, broadcastShapeTarget);
    
    CoreTensor out = new CoreTensor(CoreBinaryOps.elementwise(coreA, coreB, op));
    
    out.requiresGrad = tensorA.requiresGrad || tensorB.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensorA, tensorB);
      
      out.derivative = (grad) -> {
        if (tensorA.requiresGrad) {
          RawTensor dACore = CoreBinaryOps.elementwise(coreA, coreB, dA);
          RawTensor gradA  = CoreBinaryOps.elementwise(grad.core, dACore, (x, y) -> x * y);
          
          double[] reducedData = Engine.reduceSum(gradA.dump(), broadcastShapeTarget, tensorA.shape);
          tensorA.accumulate(new CoreTensor(reducedData, tensorA.shape));
        }
        
        if (tensorB.requiresGrad) {
          RawTensor dBCore = CoreBinaryOps.elementwise(coreA, coreB, dB);
          RawTensor gradB  = CoreBinaryOps.elementwise(grad.core, dBCore, (x, y) -> x * y);
          
          double[] reducedData = Engine.reduceSum(gradB.dump(), broadcastShapeTarget, tensorB.shape);
          tensorB.accumulate(new CoreTensor(reducedData, tensorB.shape));
        }
      };
    }
    
    return out;
  }
  
  public static CoreTensor contract(CoreTensor tensorA, CoreTensor tensorB, int[] axesA, int[] axesB) {
    CoreTensor out = new CoreTensor(CoreBinaryOps.contract(tensorA.core, tensorB.core, axesA, axesB));
    out.requiresGrad = tensorA.requiresGrad || tensorB.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensorA, tensorB);
      
      // returns the grad axes of A and B
      int[][] survivors = Shaping.getResultAxes(tensorA.shape, tensorB.shape, axesA, axesB);
      
      out.derivative = (grad) -> {
        if (tensorA.requiresGrad) {
          RawTensor gradACore = CoreBinaryOps.contract(grad.core, tensorB.core, survivors[1], axesB);
          tensorA.accumulate(new CoreTensor(gradACore));
        }
        
        if (tensorB.requiresGrad) {
          RawTensor gradBCore = CoreBinaryOps.contract(tensorA.core, grad.core, axesA, survivors[0]);
          tensorB.accumulate(new CoreTensor(gradBCore));
        }
      };
    }
    
    return out;
  }

  // ==============================================================================================
  // IMPLEMENTATION
  // ==============================================================================================

  public static CoreTensor add(CoreTensor tensorA, CoreTensor tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x + y, (x, y) -> 1.0, (x, y) -> 1.0);
  }

  public static CoreTensor sub(CoreTensor tensorA, CoreTensor tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x - y, (x, y) -> 1.0, (x, y) -> -1.0);
  }

  public static CoreTensor hadamard(CoreTensor tensorA, CoreTensor tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x * y, (x, y) -> y, (x, y) -> x);
  }

  public static CoreTensor div(CoreTensor tensorA, CoreTensor tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x / y, (x, y) -> 1.0 / y, (x, y) -> -x / Math.pow(y, 2));
  }

  public static CoreTensor pow(CoreTensor tensorA, CoreTensor tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> Math.pow(x, y), (x, y) -> y * Math.pow(x, y - 1), (x, y) -> Math.pow(x, y) * Math.log(x));
  }
}
