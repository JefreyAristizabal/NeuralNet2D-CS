using System;
using System.Text;

namespace NeuralNet
{
    public class Layer
    {
        public int InputSize, OutputSize;

        public Matrix Weights;
        public Matrix Biases;

        public Matrix Inputs;
        public Matrix Z;
        public Matrix Outputs;

        public Matrix WeightGradients;
        public Matrix BiasGradients;

        public Matrix DeltaWeights;
        public Matrix DeltaBiases;

        public ActivationType Activation;

        public Layer(int inputSize, int outputSize, ActivationType activation)
        {
            InputSize = inputSize;
            OutputSize = outputSize;
            Activation = activation;

            Weights = Matrix.Random(outputSize, inputSize, stdDev: 1.0 / Math.Sqrt(inputSize)); // Xavier
            Biases = new Matrix(outputSize, 1);

            WeightGradients = new Matrix(outputSize, inputSize);
            BiasGradients = new Matrix(outputSize, 1);
            DeltaWeights = new Matrix(outputSize, inputSize);
            DeltaBiases = new Matrix(outputSize, 1);
        }

        public Matrix FeedForward(Matrix input)
        {
            Inputs = input;
            Z = Matrix.Dot(Weights, input);
            Z.Add(Biases);
            Outputs = Z.Copy();
            Outputs.Map(GetActivation(Activation));

            if (Outputs.HasNaN()) throw new Exception("❌ NaN detectado en la salida de la capa.");
            return Outputs;
        }

        public Matrix GetActivationDerivative()
        {
            Matrix derivative = Z.Copy();
            derivative.Map(GetActivationDerivative(Activation));
            return derivative;
        }

        private Func<double, double> GetActivation(ActivationType type)
        {
            return type switch
            {
                ActivationType.Sigmoid => ActivationFunction.Sigmoid,
                ActivationType.Tanh => ActivationFunction.Tanh,
                ActivationType.ReLU => ActivationFunction.ReLU,
                ActivationType.LeakyReLU => ActivationFunction.LeakyReLU,
                _ => throw new NotImplementedException(),
            };
        }

        private Func<double, double> GetActivationDerivative(ActivationType type)
        {
            return type switch
            {
                ActivationType.Sigmoid => ActivationFunction.SigmoidDerivative,
                ActivationType.Tanh => ActivationFunction.TanhDerivative,
                ActivationType.ReLU => ActivationFunction.ReLUDerivative,
                ActivationType.LeakyReLU => ActivationFunction.LeakyReLUDerivative,
                _ => throw new NotImplementedException(),
            };
        }

        public void UpdateWeights(double learningRate)
        {
            Weights.AddScaled(WeightGradients, learningRate);
            Biases.AddScaled(BiasGradients, learningRate);
        }

        public string Serialize()
        {
            StringBuilder sb = new();

            for (int i = 0; i < Weights.Rows; i++)
                for (int j = 0; j < Weights.Cols; j++)
                    sb.AppendLine($"W {i} {j} {Weights[i, j]}");

            for (int i = 0; i < Biases.Rows; i++)
                sb.AppendLine($"B {i} {Biases[i, 0]}");

            return sb.ToString();
        }

        public void Deserialize(string[] lines)
        {
            foreach (var line in lines)
            {
                var parts = line.Split(' ');
                if (parts[0] == "W")
                {
                    int i = int.Parse(parts[1]);
                    int j = int.Parse(parts[2]);
                    Weights[i, j] = double.Parse(parts[3]);
                }
                else if (parts[0] == "B")
                {
                    int i = int.Parse(parts[1]);
                    Biases[i, 0] = double.Parse(parts[2]);
                }
            }
        }
    }
}
