
import java.util.Arrays;

import com.aufy.jnet.core.tensor.core.backend.compute.Engine;


public class backend {
  public static void main(String[] args) {
    double[] original = new double[] {1,2,3,4};
    double[] transposed = new double[] {1,3,2,4};

    System.out.println(Arrays.toString(original));
    System.out.println(Arrays.toString(Engine.makeContiguous(original, new int[] {2,2}, new int[] {1,2})));
    System.out.println("================");
    System.out.println(Arrays.toString(transposed));
  }
}
