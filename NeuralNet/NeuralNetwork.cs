using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class NeuralNetwork
    {
        private List<Layer> Layers = new List<Layer>();
        private ILoss LossFunction;
        private IOptimizer Optimizer;
        private double LearningRate = 0.001;

        public NeuralNetwork(int inputSize)
        {
            // Input size is specified when adding the first layer
        }

        public void AddLayer(int inputSize, int outputSize, ActivationType activation)
        {
            Layers.Add(new Layer(inputSize, outputSize, activation));
        }

        public void SetLossFunction(ILoss loss)
        {
            LossFunction = loss;
        }

        public void SetLearningRate(double lr)
        {
            LearningRate = lr;
        }

        public void UseSGD()
        {
            Optimizer = new SGD(LearningRate);
        }

        public void UseAdam()
        {
            Optimizer = new AdamOptimizer(LearningRate);
        }

        public double[] Predict(double[] inputArray)
        {
            Matrix a = Matrix.FromArray(inputArray);
            foreach (var layer in Layers)
                a = layer.FeedForward(a);
            return a.ToArray();
        }

        public void Train(double[][] inputs, double[][] targets, int epochs)
        {
            Console.WriteLine($"🔄 Iniciando entrenamiento por {epochs} épocas...");
            Optimizer ??= new SGD(LearningRate);
            for (int e = 1; e <= epochs; e++)
            {
                double sumErr = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    // Forward
                    var m = Matrix.FromArray(inputs[i]);
                    foreach (var L in Layers) m = L.FeedForward(m);

                    var expected = Matrix.FromArray(targets[i]);
                    sumErr += LossFunction.Calculate(m, expected);

                    // Backward
                    var error = LossFunction.Derivative(m, expected);
                    for (int j = Layers.Count - 1; j >= 0; j--)
                    {
                        var L = Layers[j];
                        var dZ = Matrix.Hadamard(error, L.GetActivationDerivative());
                        L.WeightGradients = Matrix.Dot(dZ, Matrix.Transpose(L.Inputs));
                        L.BiasGradients = dZ;

                        Optimizer.Update(L);

                        if (j > 0)
                            error = Matrix.Dot(Matrix.Transpose(Layers[j].Weights), dZ);
                    }
                }
                // Mostrar progreso cada 100 épocas
                if (e % 100 == 0 || e == 1 || e == epochs)
                    Console.WriteLine($"Epoch {e}/{epochs} – Error promedio: {sumErr / inputs.Length:F6}");
            }
        }
    }
}
