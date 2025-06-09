namespace NeuralNet
{
    public class MeanSquaredError : ILoss
    {
        public double Calculate(Matrix predicted, Matrix expected)
        {
            double sum = 0.0;
            for (int i = 0; i < predicted.Rows; i++)
            {
                for (int j = 0; j < predicted.Columns; j++)
                {
                    double diff = predicted[i, j] - expected[i, j];
                    sum += diff * diff;
                }
            }
            return sum / predicted.Rows;
        }

        public Matrix Derivative(Matrix predicted, Matrix expected)
        {
            Matrix result = new Matrix(predicted.Rows, predicted.Columns);
            for (int i = 0; i < predicted.Rows; i++)
            {
                for (int j = 0; j < predicted.Columns; j++)
                {
                    result[i, j] = 2 * (predicted[i, j] - expected[i, j]) / predicted.Rows;
                }
            }
            return result;
        }
    }
}
