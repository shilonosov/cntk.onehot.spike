using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace cntk.onehot.spike
{
    class Program
    {
        static void Main(string[] args)
        {
            const int vectorSize = 300;
            const int vocabularySize = 6000;
            const float xMax = 100f;
            const float alphaOrder = 0.75f;

            var device = DeviceDescriptor.GPUDevice(0);

            var scalarDimension = new[] {1};
            var matrixSize = new[] {vectorSize, vocabularySize};

            var iterationScalarShape = NDShape.CreateNDShape(scalarDimension);
            var iterationVectorShape = NDShape.CreateNDShape(matrixSize);
            var scalarShape = NDShape.CreateNDShape(scalarDimension);

            var coOccurrences = Variable.InputVariable(iterationScalarShape, DataType.Float, "coOccurrences - " + vocabularySize, null, false);
            var columns = Variable.InputVariable(scalarShape, DataType.Float, "columns - " + vocabularySize, null, false);
            var rows = Variable.InputVariable(scalarShape, DataType.Float, "rows - " + vocabularySize, null, false);

            var mainVectors = new Parameter(iterationVectorShape, DataType.Float, 0d, device);
            var contextVectors = new Parameter(iterationVectorShape, DataType.Float, 0d, device);
            var mainBiases = new Parameter(iterationScalarShape, DataType.Float, 0d, device);
            var contextBiases = new Parameter(iterationScalarShape, DataType.Float, 0d, device);
            var one = new Constant(iterationScalarShape, DataType.Float, 1d, device);
            var xmax = new Constant(iterationScalarShape, DataType.Float, xMax, device);
            var alpha = new Constant(iterationScalarShape, DataType.Float, alphaOrder, device);


            var weight = CNTKLib.ElementMin(one, CNTKLib.Pow(CNTKLib.ElementDivide(coOccurrences, xmax), alpha), "min");

            var oneHotRow = CNTKLib.OneHotOp(rows, vocabularySize, true, new Axis(0));
            var oneHotColumn = CNTKLib.OneHotOp(columns, vocabularySize, true, new Axis(0));

            var mainVector = CNTKLib.Times(mainVectors, oneHotColumn);
            var contextVector = CNTKLib.Times(contextVectors, oneHotRow);
            var mainBias = CNTKLib.Times(mainBiases, oneHotColumn);
            var contextBias = CNTKLib.Times(contextBiases, oneHotRow);

            var model = CNTKLib.ReduceSum(CNTKLib.TransposeTimes(mainVector, contextVector), new Axis(0));
            model = CNTKLib.Plus(model, mainBias);
            model = CNTKLib.Plus(model, contextBias);
            model = CNTKLib.Minus(model, CNTKLib.Log(coOccurrences));
            model = CNTKLib.Square(model);
            model = CNTKLib.ElementTimes(model, weight);

            var thisBatchShape = NDShape.CreateNDShape(new[] {1});

            var cooccurenceValue = Value.CreateBatch(thisBatchShape, new float[] {0.561f}, device);
            var columnsValue = Value.CreateBatch(thisBatchShape, new float[] {1f}, device);
            var rowsValue = Value.CreateBatch(thisBatchShape, new float[] {2f}, device);

            var trainDictionary = new Dictionary<Variable, Value>
            {
                {coOccurrences, cooccurenceValue},
                {columns, columnsValue},
                {rows, rowsValue}
            };

            var parameterVector = new ParameterVector(model.Parameters().ToList());

            var learner = CNTKLib.AdamLearner(
                parameterVector,
                new TrainingParameterScheduleDouble(0.1, (uint) (vocabularySize * vocabularySize)),
                new TrainingParameterScheduleDouble(0.9, (uint) (vocabularySize * vocabularySize)),
                false);

            var learners = new LearnerVector() {learner};
            var trainer = CNTKLib.CreateTrainer(model, model, model, learners);

            trainer.TrainMinibatch(trainDictionary, false, device);

            Console.WriteLine("success!");
        }
    }
}