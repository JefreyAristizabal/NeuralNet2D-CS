using NeuralNet;
using System;

//class Program
//{
//    static void Main(string[] args)
//    {
//        Console.WriteLine("🔵 Entrenando red neuronal para XOR...");

//        // Crear red neuronal: 2 entradas -> 4 neuronas ocultas -> 1 salida
//        NeuralNetwork net = new NeuralNetwork(2);

//        // ✅ Primera capa con inputSize
//        net.AddLayer(2, 4, ActivationType.Sigmoid);
//        // ✅ Capas siguientes sin inputSize
//        net.AddLayer(1, ActivationType.Sigmoid);

//        // Configurar red
//        net.SetLearningRate(0.1);
//        net.SetLossFunction(LossType.MSE);

//        // Entrenar
//        net.Train(Dataset.XOR_Inputs, Dataset.XOR_Outputs, epochs: 10000);

//        // Probar
//        Console.WriteLine("\n🧪 Resultados:");
//        for (int i = 0; i < Dataset.XOR_Inputs.Length; i++)
//        {
//            double[] output = net.Predict(Dataset.XOR_Inputs[i]);
//            Console.WriteLine($"Entrada: {Dataset.XOR_Inputs[i][0]}, {Dataset.XOR_Inputs[i][1]} => Salida: {output[0]:F4} (esperado: {Dataset.XOR_Outputs[i][0]})");
//        }

//        Console.WriteLine("\n✅ Entrenamiento completado.");

//        net.Save("modelo_xor.txt");
//        Console.WriteLine("📦 Modelo guardado en 'modelo_xor.txt'");

//        // Descomentar para probar carga
//        NeuralNetwork net2 = new NeuralNetwork(2);
//        net2.Load("modelo_xor.txt");
//    }
//}


class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("🧠 Clasificación de puntos 2D");

        // Generar datos sintéticos
        var (inputs, outputs) = Dataset2D.Generate(samplesPerClass: 500);

        // Crear red neuronal
        NeuralNetwork net = new NeuralNetwork(2);

        // Capas ocultas optimizadas
        net.AddLayer(2, 16, ActivationType.Tanh);       // Más unidades para mayor capacidad
        net.AddLayer(16, 12, ActivationType.ReLU);      // ReLU ayuda a gradientes
        net.AddLayer(12, 8, ActivationType.Tanh);       // Tanh ofrece salida centrada
        net.AddLayer(8, 1, ActivationType.Sigmoid);     // Capa de salida binaria

        // Configuraciones importantes
        net.SetLossFunction(new BinaryCrossEntropy());
        net.SetLearningRate(0.0003); // Baja tasa para estabilidad
        net.UseAdam();

        // Entrenamiento
        net.Train(inputs, outputs, epochs: 5000);

        // Evaluación de predicciones
        Console.WriteLine("\n🔍 Pruebas aleatorias:");
        Random rand = new Random();
        for (int i = 0; i < 10; i++)
        {
            int idx = rand.Next(inputs.Length);
            double[] input = inputs[idx];
            double[] prediction = net.Predict(input);
            Console.WriteLine($"Punto: ({input[0]:F2}, {input[1]:F2}) => Clase predicha: {prediction[0]:F3}");
        }

        // Evaluación global
        Console.WriteLine("\n📊 Evaluación del modelo:");
        Metrics.EvaluateBinaryClassification(net, inputs, outputs);

        // Guardar modelo si se desea
        // net.Save("modelo_2d.txt");

        // Exportar clasificación si se desea
        // Exporter.Export2DClassificationGrid(net, "clasificacion2d.csv");
    }
}