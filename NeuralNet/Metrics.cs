using System;

namespace NeuralNet
{
    public static class Metrics
    {
        public static void EvaluateBinaryClassification(NeuralNetwork net, double[][] inputs, double[][] targets)
        {
            int TP = 0, TN = 0, FP = 0, FN = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                double[] prediction = net.Predict(inputs[i]);
                int predictedClass = prediction[0] >= 0.5 ? 1 : 0;
                int actualClass = targets[i][0] >= 0.5 ? 1 : 0;

                if (predictedClass == 1 && actualClass == 1) TP++;
                else if (predictedClass == 0 && actualClass == 0) TN++;
                else if (predictedClass == 1 && actualClass == 0) FP++;
                else if (predictedClass == 0 && actualClass == 1) FN++;
            }

            double accuracy = (TP + TN) / (double)(TP + TN + FP + FN);
            double precision = TP + FP == 0 ? 0 : TP / (double)(TP + FP);
            double recall = TP + FN == 0 ? 0 : TP / (double)(TP + FN);
            double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);

            Console.WriteLine("   Accuracy : {0:P2}", accuracy);
            Console.WriteLine("   Precision: {0:P2}", precision);
            Console.WriteLine("   Recall   : {0:P2}", recall);
            Console.WriteLine("   F1 Score : {0:P2}", f1);
        }
    }
}