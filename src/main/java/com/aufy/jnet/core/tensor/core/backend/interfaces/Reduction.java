package com.aufy.jnet.core.tensor.core.backend.interfaces;
import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.tensor.core.backend.compute.Memory;
import com.aufy.jnet.core.tensor.core.backend.compute.Shaping;

@FunctionalInterface
public interface Reduction {
  double _apply(double[] inputData);
  
  /**
   * Applies a reduction to select axes of an array.
   * 
   * @param data an array of double values
   * @param shape the shape of the array
   * @param strides the strides of the array
   * @param axes the axes to reduce
   * @param operation a reduction operation to apply to the reduced subspace, can either be a lambda expression or a reference to a method
   * @param keepDims whether to keep the dimensions that are reduced (squeeze or no)
   * @return a new array containing the results of applying the operation to the data
   */
  public static double[] apply(double[] data, int[] shape, int[] strides, int[] axes, Reduction operation, boolean keepDims) {
    int[] resShape = Shaping.reduceAxes(shape, axes);
    double[] resData = new double[Reductions.prod(resShape)];
    
    int[] subShape = Shaping.reorder(shape, axes); // size of one slice
    int reductionVolume = Reductions.prod(subShape);
    
    int[] resCoords = new int[resShape.length];
    int resIdx = 0;
    
    double[] subShapeData = new double[Reductions.prod(subShape)];

    if (resShape.length == 0) {
      for (int k = 0; k < reductionVolume; k++) {
        int[] kCoords = Memory.unravel(k, subShape);
        // use mapToOffset even for scalars to account for strides/offsets
        int offset = Memory.mapToOffset(new int[0], kCoords, axes, shape, strides, true);
        subShapeData[k] = data[offset];
      }
      return new double[]{ operation._apply(subShapeData) };
    }

    do {
      for (int k = 0; k < reductionVolume; k++) { // fills up the resulting spot by looping over the slice
        int[] kCoords = Memory.unravel(k, subShape); // get the sliced indices of the kth element

        // map coords to offset coors suitable for the 1d data
        int offset = Memory.mapToOffset(resCoords, kCoords, axes, shape, strides, false);
        subShapeData[k] = data[offset];
      }

      double ans = operation._apply(subShapeData);

      resData[resIdx++] = ans;
    } while (Memory.nextCoordinate(resCoords, resShape));

    return resData;
  }
}
