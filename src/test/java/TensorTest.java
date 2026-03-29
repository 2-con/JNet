import com.aufy.jnet.Tensor;

public class TensorTest {
  public static void main(String[] args) {
    Tensor A = new Tensor(new double[] {1,2,3,4,0,5,2,1}, 2,2,2);
    Tensor B = new Tensor(new double[] {1,1,1,1}, 2,2);
    Tensor C = Tensor.matmul(A, B);

    // int[] axesA = {2};
    // int[] axesB = {1};
    // Tensor C = Tensor.contract(A, B, axesA, axesB);

    System.out.println(A);
    System.out.println(B);
    System.out.println("===================");
    System.out.println(C);
  }
}
