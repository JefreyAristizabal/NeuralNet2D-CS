namespace NeuralNet
{
    public class SGD : IOptimizer
    {
        private readonly double learningRate;

        public SGD(double learningRate)
        {
            this.learningRate = learningRate;
        }

        public void Update(Layer layer)
        {
            // pesos
            layer.Weights.AddScaled(layer.WeightGradients, -learningRate);
            // biases
            layer.Biases.AddScaled(layer.BiasGradients, -learningRate);

            // limpia gradientes
            layer.WeightGradients.Map(_ => 0);
            layer.BiasGradients.Map(_ => 0);
        }
    }
}
