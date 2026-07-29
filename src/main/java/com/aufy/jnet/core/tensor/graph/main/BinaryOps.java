package com.aufy.jnet.core.tensor.graph.main;

import java.util.List;
import java.util.function.BiFunction;

import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.arrayops.Uniform;
import com.aufy.jnet.core.backend.interfaces.BinaryOp;
import com.aufy.jnet.core.tensor.core.backend.compute.Engine;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;
import com.aufy.jnet.core.tensor.core.implementations.RawTensor;
import com.aufy.jnet.core.tensor.functional.main.RawBinaryOps;
import com.aufy.jnet.core.tensor.functional.main.RawShapeOps;

public class BinaryOps {
  /*
  dont go overboard with the additions, Tensor will do the job. just add the core stuff and Tensor will do the rest
  */

  public static CoreTensor apply(CoreTensor tensorA, CoreTensor tensorB, BiFunction<RawTensor, RawTensor, RawTensor> operation, BiFunction<RawTensor, RawTensor, RawTensor> dA, BiFunction<RawTensor, RawTensor, RawTensor> dB) {    
    CoreTensor out = new CoreTensor(operation.apply(tensorA.core, tensorB.core));
    
    out.requiresGrad = tensorA.requiresGrad || tensorB.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensorA, tensorB);
      
      out.derivative = (grad) -> {
        if (tensorA.requiresGrad) {
          RawTensor dACore = dA.apply(tensorA.core, tensorB.core);
          RawTensor gradA = RawBinaryOps.elementwise(grad.core, dACore, (x, y) -> x * y);
          tensorA.accumulate(new CoreTensor(gradA));
        }
        
        if (tensorB.requiresGrad) {
          RawTensor dBCore = dB.apply(tensorA.core, tensorB.core);
          RawTensor gradB = RawBinaryOps.elementwise(grad.core, dBCore, (x, y) -> x * y);
          tensorB.accumulate(new CoreTensor(gradB));
        }
      };
    }
    
    return out;
  }
  
  public static CoreTensor elementwise(CoreTensor tensorA, CoreTensor tensorB, BinaryOp operation, BinaryOp dA, BinaryOp dB) {
    int[] broadcastShapeTarget = Shaping.broadcastedShape(tensorA.shape, tensorB.shape);
    
    RawTensor coreA = RawShapeOps.broadcast(tensorA.core, broadcastShapeTarget);
    RawTensor coreB = RawShapeOps.broadcast(tensorB.core, broadcastShapeTarget);
    
    CoreTensor out = new CoreTensor(RawBinaryOps.elementwise(coreA, coreB, operation));
    
    out.requiresGrad = tensorA.requiresGrad || tensorB.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensorA, tensorB);
      
      out.derivative = (grad) -> {
        if (tensorA.requiresGrad) {
          RawTensor dACore = RawBinaryOps.elementwise(coreA, coreB, dA); // dy/dx
          RawTensor gradA  = RawBinaryOps.elementwise(grad.core, dACore, (x, y) -> x * y); // grad * dy/dx = grad/dx
          
          double[] reducedData = Engine.reduction(gradA.dump(), broadcastShapeTarget, tensorA.shape, (a,b) -> a + b); // reshape data if there was broadcasting
          tensorA.accumulate(new CoreTensor(reducedData, tensorA.shape));
        }
        
        if (tensorB.requiresGrad) {
          RawTensor dBCore = RawBinaryOps.elementwise(coreA, coreB, dB);
          RawTensor gradB  = RawBinaryOps.elementwise(grad.core, dBCore, (x, y) -> x * y);
          
          double[] reducedData = Engine.reduction(gradB.dump(), broadcastShapeTarget, tensorB.shape, (a,b) -> a + b);
          tensorB.accumulate(new CoreTensor(reducedData, tensorB.shape));
        }
      };
    }
    
    return out;
  }
  
  public static CoreTensor contract(CoreTensor tensorA, CoreTensor tensorB, int[] axesA, int[] axesB) {
    CoreTensor out = new CoreTensor(RawBinaryOps.contract(tensorA.core, tensorB.core, axesA, axesB));
    out.requiresGrad = tensorA.requiresGrad || tensorB.requiresGrad;
    
    if (out.requiresGrad) {
      out.parents = List.of(tensorA, tensorB);
      
      int[] freeA = Shaping.removeAxes(Tools.arrange(tensorA.rank), axesA);
      int[] freeB = Shaping.removeAxes(Tools.arrange(tensorB.rank), axesB);

      int[] freeA_inC = Tools.arrange(freeA.length); 
      int[] freeB_inC = Uniform.add(Tools.arrange(freeB.length), freeA.length);

      out.derivative = (grad) -> {
        if (tensorA.requiresGrad) {
          RawTensor gradACore = RawBinaryOps.contract(tensorB.core, grad.core, freeB, freeB_inC);

          int[] rawAxes = Tools.concat(axesA, freeA);
          int[] permuteOrder = new int[tensorA.rank];

          for (int i = 0; i < rawAxes.length; i++) {
            permuteOrder[rawAxes[i]] = i;
          }

          tensorA.accumulate(ShapeOps.permute(new CoreTensor(gradACore), permuteOrder));
        }
        if (tensorB.requiresGrad) {
          RawTensor gradBCore = RawBinaryOps.contract(grad.core, tensorA.core, freeA_inC, freeA);
          int[] rawAxes = Tools.concat(freeB, axesB);
          int[] permuteOrder = new int[tensorB.rank];

          for (int i = 0; i < rawAxes.length; i++) {
            permuteOrder[rawAxes[i]] = i;
          }

          tensorB.accumulate(ShapeOps.permute(new CoreTensor(gradBCore), permuteOrder));
        }
      };
    }
    
    return out;
  }

  public static CoreTensor add(CoreTensor tensorA, CoreTensor tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x + y, (x, y) -> 1.0, (x, y) -> 1.0);
  }
}
