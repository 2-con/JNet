package com.aufy.jnet.tensor.core.backend.func;
import com.aufy.jnet.tensor.core.backend.compute.PointerLogic;
import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.backend.util.ArrayTools;

@FunctionalInterface
public interface Reduction {
  double reduce(double[] accumulator);
  
  public static double[] apply(double[] data, int[] shape, int[] strides, int[] axes, Reduction operation, boolean keepDims) {
    int[] resShape = Shaping.getSurvivors(shape, axes);
    double[] resData = new double[ArrayTools.prod(resShape)];
    
    int[] subShape = Shaping.getSubShape(shape, axes); // size of one slice
    int reductionVolume = ArrayTools.prod(subShape);
    
    int[] resCoords = new int[resShape.length];
    int resIdx = 0;
    
    double[] subShapeData = new double[ArrayTools.prod(subShape)];

    if (resShape.length == 0) {
      for (int k = 0; k < reductionVolume; k++) {
        int[] kCoords = Shaping.unravel(k, subShape);
        // use mapToOffset even for scalars to account for strides/offsets
        int offset = PointerLogic.mapToOffset(new int[0], kCoords, axes, shape, strides, true);
        subShapeData[k] = data[offset];
      }
      return new double[]{ operation.reduce(subShapeData) };
    }

    do {
      for (int k = 0; k < reductionVolume; k++) { // fills up the resulting spot by looping over the slice
        int[] kCoords = Shaping.unravel(k, subShape); // get the sliced indices of the kth element

        // map coords to offset coors suitable for the 1d data
        int offset = PointerLogic.mapToOffset(resCoords, kCoords, axes, shape, strides, true);
        subShapeData[k] = data[offset];
      }

      double ans = operation.reduce(subShapeData);

      resData[resIdx++] = ans;
    } while (PointerLogic.nextCoordinate(resCoords, resShape));

    return resData;
  }
}
