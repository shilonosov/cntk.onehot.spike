﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reactive.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace cntk.onehot.spike
{
    class Program
    {
        static void PrintDim(Function source, string name)
        {
            var dimensionStrings = source.Output
                .Shape
                .Dimensions
                .Select((x, i) => $"dim{i}: {x}");
            var dimensions = string.Join(", ", dimensionStrings);
            Console.WriteLine($"{name}: {dimensions}");
        }

        static void Main(string[] args)
        {
            const int vectorSize = 300;
            const int vocabularySize = 6000;
            const float xMax = 100f;
            const float alphaOrder = 0.75f;

            var device = DeviceDescriptor.CPUDevice;

            var scalarDimension = new[] {1, 1 };
            var matrixSize = new[] {vectorSize, vocabularySize};
            var vectorDimension = new[] {1, vocabularySize};

            var iterationScalarShape = NDShape.CreateNDShape(scalarDimension);
            var iterationMatrixShape = NDShape.CreateNDShape(matrixSize);
            var iterationVectorShape = NDShape.CreateNDShape(vectorDimension);

            var oneHotShape = NDShape.CreateNDShape(new[] {vocabularySize, 1});

            var coOccurrences = Variable.InputVariable(new int[] { 1 }, DataType.Float, "coOccurrences - " + vocabularySize, null, false);
            var columns = Variable.InputVariable(new int[] { 1 }, DataType.Float, "columns - " + vocabularySize, null, false);
            var rows = Variable.InputVariable(new int[]{1}, DataType.Float, "rows - " + vocabularySize, null, false);

            var mainVectors = new Parameter(iterationMatrixShape, DataType.Float, 0d, device);
            var contextVectors = new Parameter(iterationMatrixShape, DataType.Float, 0d, device);
            var mainBiases = new Parameter(iterationVectorShape, DataType.Float, 0d, device);
            var contextBiases = new Parameter(iterationVectorShape, DataType.Float, 0d, device);
            var one = new Constant(iterationScalarShape, DataType.Float, 1d, device);
            var xmax = new Constant(iterationScalarShape, DataType.Float, xMax, device);
            var alpha = new Constant(iterationScalarShape, DataType.Float, alphaOrder, device);


            var weight = CNTKLib.ElementMin(one, CNTKLib.Pow(CNTKLib.ElementDivide(coOccurrences, xmax), alpha), "min");

            var oneHotRow = CNTKLib.OneHotOp(rows, vocabularySize, true, new Axis(0));
            PrintDim(oneHotRow, nameof(oneHotRow));
            var oneHotColumn = CNTKLib.OneHotOp(columns, vocabularySize, true, new Axis(0));

            var mainVector = CNTKLib.Alias(CNTKLib.Times(mainVectors, oneHotColumn));
            PrintDim(mainVector, nameof(mainVector));
            var contextVector = CNTKLib.Alias(CNTKLib.Times(contextVectors, oneHotRow));
            PrintDim(contextVector, nameof(contextVector));
            var mainBias = CNTKLib.Alias(CNTKLib.Times(mainBiases, oneHotColumn));
            PrintDim(mainBias, nameof(mainBias));
            var contextBias = CNTKLib.Alias(CNTKLib.Times(contextBiases, oneHotRow));
            PrintDim(contextBias, nameof(contextBias));

            var model = CNTKLib.TransposeTimes(mainVector, contextVector);
            PrintDim(model, "CNTKLib.Times(mainVector, contextVector)");
            model = CNTKLib.Plus(model, mainBias);
            PrintDim(model, "CNTKLib.Plus(model, mainBias)");
            model = CNTKLib.Plus(model, contextBias);
            PrintDim(model, "CNTKLib.Plus(model, contextBias)");
            model = CNTKLib.Minus(model, CNTKLib.Log(coOccurrences));
            PrintDim(model, "CNTKLib.Minus(model, CNTKLib.Log(coOccurrences))");
            model = CNTKLib.Square(model);
            PrintDim(model, "CNTKLib.Square(model)");
            model = CNTKLib.ElementTimes(model, weight);
            PrintDim(model, "CNTKLib.ElementTimes(model, weight)");

            var thisBatchShape = NDShape.CreateNDShape(new[] {1});


            var parameterVector = new ParameterVector(model.Parameters().ToList());

            //var learner = CNTKLib.AdamLearner(
            //    parameterVector,
            //    new TrainingParameterScheduleDouble(0.1, (uint) (vocabularySize * vocabularySize)),
            //    new TrainingParameterScheduleDouble(0.9, (uint) (vocabularySize * vocabularySize)),
            //    false);

            var learner = CNTKLib.SGDLearner(
                parameterVector,
                new TrainingParameterScheduleDouble(0.1, (uint)(vocabularySize * vocabularySize)));

            var learners = new LearnerVector() {learner};
            var trainer = CNTKLib.CreateTrainer(model, model, model, learners);



            var count = (int)(vocabularySize*vocabularySize*0.2d*0.2d);
            var floats = GetRandomFloats(count).ToArray();
            var fColumns = GetRandomInts(count, 0, vocabularySize).ToArray();
            var fRows = GetRandomInts(count, 0, vocabularySize).ToArray();

            var all = floats.Zip(fColumns, (f, c) => (f: f, c: c)).Zip(fRows, (tuple, r) => (tuple.c, tuple.c, r))
                .ToObservable()
                .Buffer(10000)
                .Select(x => (f: x.Select(y => y.Item1).ToArray(), c: x.Select(y => y.Item2).ToArray(), r: x.Select(y => y.Item3).ToArray()))
                .ToArray()
                .Wait();

            Console.WriteLine($"count: {count}");

            var stopwatch = new Stopwatch();
            stopwatch.Start();
            for (var e = 0; e < 1; e++)
            {
                for (var i = 0; i < all.Length; i++)
                {
                    var valueTuple = all[i];

                    var cooccurenceValue = Value.CreateBatch(thisBatchShape, valueTuple.f, device);
                    var columnsValue = Value.CreateBatch(thisBatchShape, valueTuple.c, device);
                    var rowsValue = Value.CreateBatch(thisBatchShape, valueTuple.r, device);

                    var trainDictionary = new Dictionary<Variable, Value>
                    {
                        {coOccurrences, cooccurenceValue},
                        {columns, columnsValue},
                        {rows, rowsValue}
                    };


                    trainer.TrainMinibatch(trainDictionary, false, device);

                    if (i % 100 == 0)
                    {
                        Console.WriteLine($"e: {e}\ti: {stopwatch.Elapsed:g}");
                    }
                }
            }
            stopwatch.Stop();


            Console.WriteLine($"success: {stopwatch.Elapsed:g}");
        }

        public static IEnumerable<float> GetRandomFloats(int count)
        {
            var random = new Random();
            for(var i = 0; i < count; ++i)
            {
                yield return (float)random.NextDouble();
            }
        }

        public static IEnumerable<float> GetRandomInts(int count, int min, int max)
        {
            var random = new Random();
            for (var i = 0; i < count; ++i)
            {
                yield return (float)random.Next(min, max);
            }
        }
    }
}