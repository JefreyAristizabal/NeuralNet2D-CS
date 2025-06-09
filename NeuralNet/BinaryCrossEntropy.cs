namespace NeuralNet
{
    public class BinaryCrossEntropy : ILoss
    {
        public double Calculate(Matrix predicted, Matrix expected)
        {
            double epsilon = 1e-12;
            double loss = 0.0;

            for (int i = 0; i < predicted.Rows; i++)
            {
                for (int j = 0; j < predicted.Columns; j++)
                {
                    double y = expected[i, j];
                    double yHat = Math.Min(1.0 - epsilon, Math.Max(epsilon, predicted[i, j]));

                    loss += -y * Math.Log(yHat) - (1 - y) * Math.Log(1 - yHat);
                }
            }

            return loss / predicted.Rows;
        }

        public Matrix Derivative(Matrix predicted, Matrix expected)
        {
            double epsilon = 1e-12;
            Matrix result = new Matrix(predicted.Rows, predicted.Columns);

            for (int i = 0; i < predicted.Rows; i++)
            {
                for (int j = 0; j < predicted.Columns; j++)
                {
                    double y = expected[i, j];
                    double yHat = Math.Min(1.0 - epsilon, Math.Max(epsilon, predicted[i, j]));

                    result[i, j] = (-y / yHat) + ((1 - y) / (1 - yHat));
                }
            }

            return result;
        }
    }
}
