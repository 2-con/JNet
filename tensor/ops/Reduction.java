package tensor.ops;
import tensor.core.Engine;

@FunctionalInterface
public interface Reduction {
  double reduce(double[] accumulator);
  
  public static double[] apply(double[] data, int[] shape, int[] strides, int[] axes, Reduction operation) {
    int[] resShape = Engine.getSurvivors(shape, axes);
    double[] resData = new double[Engine.sizeOf(resShape)];
    
    int[] subShape = Engine.getSubShape(shape, axes); // size of one slice
    int reductionVolume = Engine.sizeOf(subShape);

    int[] resCoords = new int[resShape.length];
    int resIdx = 0;

    double[] subShapeData = new double[Engine.sizeOf(subShape)];

    do {
      for (int k = 0; k < reductionVolume; k++) { // fills up the resulting spot by looping over the slice
        int[] kCoords = Engine.unravel(k, subShape); // get the sliced indices of the kth element

        // map coords to offset coors suitable for the 1d data
        int offset = Engine.mapToOffset(resCoords, kCoords, axes, shape, strides, true);
        subShapeData[k] = data[offset];
      }

      double ans = operation.reduce(subShapeData);

      resData[resIdx++] = ans;
    } while (Engine.nextCoordinate(resCoords, resShape));

    return resData;
  }
}
