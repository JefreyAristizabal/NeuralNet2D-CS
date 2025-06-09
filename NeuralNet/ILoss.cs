namespace NeuralNet
{
    public interface ILoss
    {
        double Calculate(Matrix predicted, Matrix expected);
        Matrix Derivative(Matrix predicted, Matrix expected);
    }
}


