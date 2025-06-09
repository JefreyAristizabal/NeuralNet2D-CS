namespace NeuralNet
{
    public interface IOptimizer
    {
        /// <summary>
        /// Actualiza pesos y biases de la capa, usando layer.WeightGradients y layer.BiasGradients.
        /// </summary>
        void Update(Layer layer);
    }
}
