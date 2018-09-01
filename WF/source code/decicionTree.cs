using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Accord.Statistics.Models.Regression.Linear;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using Accord.MachineLearning.VectorMachines;
using Accord.Math.Distances;

namespace DataSetsSparsity
{
    class decicionTree
    {
        private double[][] training_dt;
        private long[][] training_GridIndex_dt;
        private double[][] training_label;
        private bool[] Dime2Take;

        private class SplitData
        {
            public double error;
            public GeoWave child0;
            public GeoWave child1;

            public SplitData(double error, GeoWave child0, GeoWave child1)
            {
                this.error = error;
                this.child0 = child0;
                this.child1 = child1;
            }
        }

        public decicionTree( DB db, bool[] Dime2Take)
        {
            this.training_dt = db.training_dt;
            this.training_label = db.training_label;
            this.training_GridIndex_dt = db.DBtraining_GridIndex_dt;
            this.Dime2Take = Dime2Take;
        }

        public List<GeoWave> getdecicionTree(List<int> trainingArr, int[][] boundingBox, int seed = -1)
        {
            //CREATE DECISION_GEOWAVEARR
            List<GeoWave> decision_GeoWaveArr = new List<GeoWave>();

            //SET ROOT WAVELETE
            GeoWave gwRoot = new GeoWave(training_dt[0].Count(), training_label[0].Count());

            //SET REGION POINTS IDS
            gwRoot.pointsIdArray = trainingArr;
            boundingBox.CopyTo(gwRoot.isotropicSplitsParameters.boundingBox, 0);

            decision_GeoWaveArr.Add(gwRoot);
            DecomposeWaveletsByConsts(decision_GeoWaveArr, seed);

            //SET ID
            for (int i = 0; i < decision_GeoWaveArr.Count; i++)
                decision_GeoWaveArr[i].ID = i;

            //get sorted list
            decision_GeoWaveArr = decision_GeoWaveArr.OrderByDescending(o => o.norm).ToList();

            return decision_GeoWaveArr;
        }

        public void DecomposeWaveletsByConsts(List<GeoWave> GeoWaveArr, int seed = -1)//SHOULD GET LIST WITH ROOT GEOWAVE
        {
            GeoWaveArr[0].MeanValue = GeoWaveArr[0].calc_MeanValue(training_label, GeoWaveArr[0].pointsIdArray);
            GeoWaveArr[0].computeNormOfConsts(Convert.ToDouble(userConfig.partitionType));
            GeoWaveArr[0].level = 0;

            if (seed == -1)
                recursiveBSP_WaveletsByConsts(GeoWaveArr, 0);
            else recursiveBSP_WaveletsByConsts(GeoWaveArr, 0, seed);//0 is the root index
        }

        private void doSVMRegressopSplit(GeoWave currentWave, Dictionary<SplitType, SplitData> splitsData, double currentError,
            bool[] Dim2TakeNode)
        {
            double lowestError = currentError;
            double[] bestSvmApprox = new double[training_dt.Length];
            for (int labelIdx = 0; labelIdx < training_label[0].Length; labelIdx++)
            {
                double[] svmApprox = new double[currentWave.pointsIdArray.Count];
                SupportVectorMachine<Linear> svmRegression = getSVMRegression(currentWave, labelIdx, Dim2TakeNode, ref svmApprox);
                if (null != svmRegression)
                {
                    double[] tmpSvmApporx = new double[training_dt.Length];
                    for (int i = 0; i < currentWave.pointsIdArray.Count; i++)
                    {
                        int index = currentWave.pointsIdArray[i];
                        tmpSvmApporx[index] = svmApprox[i];
                    }

                    double svmSplitValue = 0;
                    double error = getBestSVMRegressionSplit(currentWave, tmpSvmApporx, ref svmSplitValue);
                    if (error < lowestError)
                    {
                        lowestError = error;
                        currentWave.svmRegressionSplitsParameters.svmRegression = svmRegression;
                        currentWave.svmRegressionSplitsParameters.labelIdx = labelIdx;
                        currentWave.svmRegressionSplitsParameters.svmRegressionSplitValue = svmSplitValue;
                        currentWave.svmRegressionSplitsParameters.Dim2TakeNode = Dim2TakeNode;
                        bestSvmApprox = tmpSvmApporx;
                    }
                }
                
            }

            if (lowestError >= currentError)
            {
                return;
            }

            GeoWave child0 = new GeoWave(currentWave.isotropicSplitsParameters.boundingBox, training_label[0].Count());
            GeoWave child1 = new GeoWave(currentWave.isotropicSplitsParameters.boundingBox, training_label[0].Count());

            setChildrensPointsAndMeanValueSVMRegression(child0, child1, currentWave, bestSvmApprox);
            splitsData.Add(SplitType.SVM_REGRESSION_SPLITS, new SplitData(lowestError, child0, child1));
        }

        private void doLinearRegressionSplit(GeoWave currentWave, Dictionary<SplitType, SplitData> splitsData, double currentError,
            bool[] Dim2TakeNode)
        {
            bool isSplitOk = getBestLinearRegressionSplitResult(currentWave, Dim2TakeNode);
            if (!isSplitOk)
            {
                return;
            }

            currentWave.linearRegressionSplitsParameters.Dim2TakeNode = Dim2TakeNode;
            GeoWave child0 = new GeoWave(currentWave.isotropicSplitsParameters.boundingBox, training_label[0].Count());
            GeoWave child1 = new GeoWave(currentWave.isotropicSplitsParameters.boundingBox, training_label[0].Count());

            double error = setChildrensPointsAndMeanValueLinearRegression(ref child0, ref child1, currentWave);
            if (error >= currentError)
            {
                return;
            }
            splitsData.Add(SplitType.LINEAR_REGRESSION_SPLITS, new SplitData(error, child0, child1));
        }

        private void doIsotropicSplit(GeoWave currentWave, Dictionary<SplitType, SplitData> splitsData,
            double currentError, bool[] Dim2TakeNode)
        {
            int dimIndex = -1;
            int Maingridindex = -1;
            double error = double.MaxValue;
            bool isSplitOk = getBestPartitionResult(ref dimIndex, ref Maingridindex, ref error, currentWave, currentError, Dim2TakeNode);
            if (!isSplitOk)
            {
                for (int i = 0; i < Dim2TakeNode.Count(); i++)
                    Dim2TakeNode[i] = (Dim2TakeNode[i] == true) ? false : true;
                isSplitOk = getBestPartitionResult(ref dimIndex, ref Maingridindex, ref error, currentWave, currentError, Dim2TakeNode);
                if (!isSplitOk)
                    return;
            }

            GeoWave child0 = new GeoWave(currentWave.isotropicSplitsParameters.boundingBox, training_label[0].Count());
            GeoWave child1 = new GeoWave(currentWave.isotropicSplitsParameters.boundingBox, training_label[0].Count());

            //set partition
            child0.isotropicSplitsParameters.boundingBox[1][dimIndex] = Maingridindex;
            child1.isotropicSplitsParameters.boundingBox[0][dimIndex] = Maingridindex;

            //DOCUMENT ON CHILDREN
            child0.isotropicSplitsParameters.dimIndex = dimIndex;
            child0.isotropicSplitsParameters.maingridIndex = Maingridindex;
            child1.isotropicSplitsParameters.dimIndex = dimIndex;
            child1.isotropicSplitsParameters.maingridIndex = Maingridindex;
            child0.isotropicSplitsParameters.maingridValue = wf.Program.MainGrid[dimIndex][Maingridindex];
            child1.isotropicSplitsParameters.maingridValue = wf.Program.MainGrid[dimIndex][Maingridindex];

            //DOCUMENT ON PARENT
            currentWave.isotropicSplitsParameters.dimIndexSplitter = dimIndex;
            currentWave.isotropicSplitsParameters.splitValue = wf.Program.MainGrid[dimIndex][Maingridindex];
            
            if (wf.Program.IsBoxSingular(child0.isotropicSplitsParameters.boundingBox, training_dt[0].Count()) || wf.Program.IsBoxSingular(child1.isotropicSplitsParameters.boundingBox, training_dt[0].Count()))
                return;

            //SHOULD I VERIFY THAT THE CHILD IS NOT ITS PARENT ? (IN CASES WHERE CAN'T MODEFY THE PARTITION)

            setChildrensPointsAndMeanValue(ref child0, ref child1, dimIndex, currentWave.pointsIdArray);

            splitsData.Add(SplitType.REGULAR_ISOTROPIC_SPLITS, new SplitData(error, child0, child1));
        }

        private void doSvmClassificationSplit(GeoWave currentWave, Dictionary<SplitType, SplitData> splitsData, double currentError,
            bool[] Dim2TakeNode)
        {
            bool isSplitOk = getBestSvmClassificationSplit(currentWave, Dim2TakeNode);
            if (!isSplitOk)
            {
                return;
            }

            currentWave.svmClassificationSplitParameters.Dim2TakeNode = Dim2TakeNode;
            GeoWave child0 = new GeoWave(currentWave.isotropicSplitsParameters.boundingBox, training_label[0].Count());
            GeoWave child1 = new GeoWave(currentWave.isotropicSplitsParameters.boundingBox, training_label[0].Count());

            double error = setChildrensPointsAndMeanValueSvmClassification(ref child0, ref child1, currentWave);
            if (error >= currentError)
            {
                return;
            }
            splitsData.Add(SplitType.SVM_CLASSIFICATION_SPLITS, new SplitData(error, child0, child1));
        }

        private void recursiveBSP_WaveletsByConsts(List<GeoWave> GeoWaveArr, int GeoWaveID, int seed=0)
        {
            //CALC APPROX_SOLUTION FOR GEO WAVE
            double Error = GeoWaveArr[GeoWaveID].calc_MeanValueReturnError(training_label, GeoWaveArr[GeoWaveID].pointsIdArray);
            if (Error < userConfig.approxThresh || GeoWaveArr[GeoWaveID].pointsIdArray.Count() <= userConfig.minNodeSize || userConfig.boundLevelDepth <=  GeoWaveArr[GeoWaveID].level)
                return;

            var ran1 = new Random(seed);
            var ran2 = new Random(GeoWaveID);
            int one = ran1.Next(0, int.MaxValue / 10);
            int two = ran2.Next(0, int.MaxValue / 10);
            bool[] Dim2TakeNode = getDim2Take(one + two);

            Dictionary<SplitType, SplitData> splitsData = new Dictionary<SplitType, SplitData>();

            GeoWave currentWave = GeoWaveArr[GeoWaveID];

            if (userConfig.useSVMRegression)
            {                
                doSVMRegressopSplit(currentWave, splitsData, Error, Dim2TakeNode);
            }
            if (userConfig.useLinearRegression)
            {
                doLinearRegressionSplit(currentWave, splitsData, Error, Dim2TakeNode);
            }
            if (userConfig.useIsotropicSplits)
            {
                doIsotropicSplit(currentWave, splitsData, Error, Dim2TakeNode);
            }
            if (userConfig.useSVMClassification)
            {
                doSvmClassificationSplit(currentWave, splitsData, Error, Dim2TakeNode);
            }

            SplitType bestSplitType = SplitType.NO_SPLIT;
            double lowestError = Error;
            foreach (SplitType splitType in splitsData.Keys)
            {
                SplitData splitData = splitsData[splitType];
                if (splitData.error < lowestError)
                {
                    lowestError = splitData.error;
                    bestSplitType = splitType;
                }
            }

            if (SplitType.NO_SPLIT == bestSplitType)
            {
                return;
            }
            currentWave.splitType = bestSplitType;

            GeoWave child0 = splitsData[bestSplitType].child0;
            GeoWave child1 = splitsData[bestSplitType].child1;


            //SET TWO CHILDS
            child0.parentID = child1.parentID = GeoWaveID;
            child0.child0 = child1.child0 = -1;
            child0.child1 = child1.child1 = -1;
            child0.level = child1.level = currentWave.level + 1;

            child0.computeNormOfConsts(currentWave, Convert.ToDouble(userConfig.partitionType));
            child1.computeNormOfConsts(currentWave, Convert.ToDouble(userConfig.partitionType));
            GeoWaveArr.Add(child0);
            GeoWaveArr.Add(child1);
            currentWave.child0 = GeoWaveArr.Count - 2;
            currentWave.child1 = GeoWaveArr.Count - 1;

            //RECURSION STEP !!!
            recursiveBSP_WaveletsByConsts(GeoWaveArr, currentWave.child0, seed);
            recursiveBSP_WaveletsByConsts(GeoWaveArr, currentWave.child1, seed);

        }

        private bool getBestPartitionResult(ref int dimIndex, ref int Maingridindex, ref double error, GeoWave currentWave, double Error, bool[] Dims2Take)
        {
            double[][] error_dim_partition = new double[2][];//error, maingridIndex
            error_dim_partition[0] = new double[training_dt[0].Count()];
            error_dim_partition[1] = new double[training_dt[0].Count()];

            //PARALLEL RUN - SEARCHING BEST PARTITION IN ALL DIMS
            if (userConfig.useParallel)
            {
                Parallel.For(0, training_dt[0].Count(), i =>
                {
                    //double[] tmpResult = getBestPartition(i, GeoWaveArr[GeoWaveID]);
                    if (Dims2Take[i])
                    {
                        double[] tmpResult = getBestPartitionLargeDB(i, currentWave);
                        error_dim_partition[0][i] = tmpResult[0];//error
                        error_dim_partition[1][i] = tmpResult[1];//maingridIndex                    
                    }
                    else
                    {
                        error_dim_partition[0][i] = double.MaxValue;//error
                        error_dim_partition[1][i] = -1;//maingridIndex                    
                    }
                });
            }
            else
            {
                for (int i = 0; i < training_dt[0].Count(); i++)
                {
                    //double[] tmpResult = getBestPartition(i, GeoWaveArr[GeoWaveID]);
                    if (Dims2Take[i])
                    {
                        double[] tmpResult = getBestPartitionLargeDB(i, currentWave);
                        error_dim_partition[0][i] = tmpResult[0];//error
                        error_dim_partition[1][i] = tmpResult[1];//maingridIndex                    
                    }
                    else
                    {
                        error_dim_partition[0][i] = double.MaxValue;//error
                        error_dim_partition[1][i] = -1;//maingridIndex                    
                    }
                }
            }

            dimIndex = Enumerable.Range(0, error_dim_partition[0].Count())
                .Aggregate((a, b) => (error_dim_partition[0][a] < error_dim_partition[0][b]) ? a : b);

            error = error_dim_partition[0][dimIndex];

            if (error >= Error)
                return false;//if best partition doesn't help - return

            Maingridindex = Convert.ToInt32(error_dim_partition[1][dimIndex]);
            return true;
        }

        private SupportVectorMachine<Linear> getSVMRegression(GeoWave geoWave, int labelIdx, bool[] Dim2TakeNode, ref double[] svmApprox)
        {
            SupportVectorMachine<Linear> svmRegression = null;
            double[][] dataForRegression = new double[geoWave.pointsIdArray.Count][];
            double[] labelForRegression = new double[geoWave.pointsIdArray.Count];
            int amountOfFeatures = training_dt[0].Length;
            for (int i = 0; i < geoWave.pointsIdArray.Count; i++)
            {
                int index = geoWave.pointsIdArray[i];
                dataForRegression[i] = new double[userConfig.nFeatures];
                int k = 0;
                for (int j = 0; j < amountOfFeatures; j++)
                {
                    if (Dim2TakeNode[j])
                    {
                        dataForRegression[i][k] = training_dt[index][j];
                        k++;
                    }                    
                }
                labelForRegression[i] = training_label[index][labelIdx];
            }

            LinearRegressionNewtonMethod tmpSvmRegression = new LinearRegressionNewtonMethod()
            {
                UseComplexityHeuristic = true
            };

            try
            {
                svmRegression = tmpSvmRegression.Learn(dataForRegression, labelForRegression);
                svmApprox = svmRegression.Score(dataForRegression);
            }
            catch (Exception e)
            {
                return null;
            }
            if (svmApprox.Contains(double.NaN))
            {
                return null;
            }
            return svmRegression;
        }

        private double getBestSVMRegressionSplit(GeoWave geoWave, double[] tmpSvmApprox, ref double svmSplitValue)
        {
            List<int> tmpIDs = new List<int>(geoWave.pointsIdArray);
            tmpIDs.Sort(delegate (int c1, int c2) { return tmpSvmApprox[c1].CompareTo(tmpSvmApprox[c2]); });

            if (tmpSvmApprox[tmpIDs[0]] == tmpSvmApprox[tmpIDs[tmpIDs.Count - 1]])//all values are the same 
            {
                return double.MaxValue;
            }

            int best_ID = -1;
            double lowest_err = double.MaxValue;
            double[] leftAvg = new double[geoWave.MeanValue.Count()];
            double[] rightAvg = new double[geoWave.MeanValue.Count()];
            double[] leftErr = geoWave.calc_MeanValueReturnError(training_label, geoWave.pointsIdArray, ref leftAvg);//CONTAINES ALL POINTS - AT THE BEGINING
            double[] rightErr = new double[geoWave.MeanValue.Count()];


            double N_points = Convert.ToDouble(tmpIDs.Count);
            double tmp_err;


            for (int i = 0; i < tmpIDs.Count - 1; i++)//we dont calc the last (rightmost) boundary - it equal to the left most
            {
                tmp_err = 0;
                for (int j = 0; j < geoWave.MeanValue.Count(); j++)
                {
                    leftErr[j] = leftErr[j] - (N_points - i) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - leftAvg[j]) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - leftAvg[j]) / (N_points - i - 1);
                    leftAvg[j] = (N_points - i) * leftAvg[j] / (N_points - i - 1) - training_label[tmpIDs[tmpIDs.Count - i - 1]][j] / (N_points - i - 1);
                    rightErr[j] = rightErr[j] + (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - rightAvg[j]) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - rightAvg[j]) * Convert.ToDouble(i) / Convert.ToDouble(i + 1);
                    rightAvg[j] = rightAvg[j] * Convert.ToDouble(i) / Convert.ToDouble(i + 1) + training_label[tmpIDs[tmpIDs.Count - i - 1]][j] / Convert.ToDouble(i + 1);
                    tmp_err += leftErr[j] + rightErr[j];

                }
                if (lowest_err > tmp_err && tmpSvmApprox[tmpIDs[tmpIDs.Count - i - 1]] != tmpSvmApprox[tmpIDs[tmpIDs.Count - i - 2]]
                    && (i + 1) >= userConfig.minNodeSize && (i + userConfig.minNodeSize) < tmpIDs.Count)
                {
                    best_ID = tmpIDs[tmpIDs.Count - i - 1];
                    lowest_err = tmp_err;
                }
            }

            if (best_ID == -1)
            {
                return double.MaxValue;
            }

            svmSplitValue = tmpSvmApprox[best_ID];

            return Math.Max(lowest_err, 0);
        }

        private bool getBestLinearRegressionSplitResult(GeoWave geoWave, bool[] Dim2TakeNode)
        {
            double[][] dataForRegression = new double[geoWave.pointsIdArray.Count][];
            double[][] labelForRegression = new double[geoWave.pointsIdArray.Count][];
            int amountOfFeatures = training_dt[0].Length;
            int amountOfLables = training_label[0].Length;
            for (int i = 0; i < geoWave.pointsIdArray.Count; i++)
            {
                int index = geoWave.pointsIdArray[i];
                dataForRegression[i] = new double[userConfig.nFeatures];
                labelForRegression[i] = new double[amountOfLables];
                int k = 0;
                for (int j = 0; j < amountOfFeatures; j++)
                {
                    if (Dim2TakeNode[j])
                    {
                        dataForRegression[i][k] = training_dt[index][j];
                        k++;
                    }
                                      
                }
                for (int j = 0; j < amountOfLables; j++)
                {
                    labelForRegression[i][j] = training_label[index][j] - geoWave.MeanValue[j];
                }
            }
            OrdinaryLeastSquares ols = new OrdinaryLeastSquares();
            try
            {
                geoWave.linearRegressionSplitsParameters.linearRegression = ols.Learn(dataForRegression, labelForRegression);
            }
            catch(Exception e)
            {
                return false;
            }
            return true;
            
        }

        private bool getBestSvmClassificationSplit(GeoWave geoWave, bool[] Dim2TakeNode)
        {
            double[][] dataForSvm = new double[geoWave.pointsIdArray.Count][];
            int[] labelForSvm = new int[geoWave.pointsIdArray.Count];
            int amountOfFeatures = training_dt[0].Length;
            for (int i = 0; i < geoWave.pointsIdArray.Count; i++)
            {
                int index = geoWave.pointsIdArray[i];
                dataForSvm[i] = new double[userConfig.nFeatures];
                int k = 0;
                for (int j = 0; j < amountOfFeatures; j++)
                {
                    if (Dim2TakeNode[j])
                    {
                        dataForSvm[i][k] = training_dt[index][j];
                        k++;
                    }
                }

                labelForSvm[i] = (int)training_label[index][0];
            }

            // Create a one-vs-one multi-class SVM learning algorithm 
            SequentialMinimalOptimization<Gaussian> tmpSvm = new SequentialMinimalOptimization<Gaussian>()
            {
                UseComplexityHeuristic = true,
                UseKernelEstimation = true
            };

            try
            {
                geoWave.svmClassificationSplitParameters.svm = tmpSvm.Learn(dataForSvm, labelForSvm);
            }
            catch (Exception e)
            {
                return false;
            }

            return true;
        }

        private double[] getBestPartitionLargeDB(int dimIndex, GeoWave geoWave)
        {
            double[] error_n_point = new double[2];//error index
            if (wf.Program.MainGrid[dimIndex].Count == 1)//empty feature
            {
                error_n_point[0] = double.MaxValue;
                error_n_point[1] = -1;
                return error_n_point;
            }
            //sort ids (for labels) acording to position at Form1.MainGrid[dimIndex][index]
            List<int> tmpIDs = new List<int>(geoWave.pointsIdArray);
            tmpIDs.Sort(delegate(int c1, int c2) { return training_dt[c1][dimIndex].CompareTo(training_dt[c2][dimIndex]); });

            if (training_dt[tmpIDs[0]][dimIndex] == training_dt[tmpIDs[tmpIDs.Count - 1]][dimIndex])//all values are the same 
            {
                error_n_point[0] = double.MaxValue;
                error_n_point[1] = -1;
                return error_n_point;
            }

            int best_ID = -1;
            double lowest_err = double.MaxValue;
            double[] leftAvg = new double[geoWave.MeanValue.Count()];
            double[] rightAvg = new double[geoWave.MeanValue.Count()];
            double[] leftErr = geoWave.calc_MeanValueReturnError(training_label, geoWave.pointsIdArray, ref leftAvg);//CONTAINES ALL POINTS - AT THE BEGINING
            double[] rightErr = new double[geoWave.MeanValue.Count()];

            double N_points = Convert.ToDouble(tmpIDs.Count);

            double tmp_err;

            for (int i = 0; i < tmpIDs.Count - 1; i++)//we dont calc the last (rightmost) boundary - it equal to the left most
            {
                tmp_err = 0;
                for (int j = 0; j < geoWave.MeanValue.Count(); j++)
                {
                    leftErr[j] = leftErr[j] - (N_points - i) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - leftAvg[j]) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - leftAvg[j]) / (N_points - i - 1);
                    leftAvg[j] = (N_points - i) * leftAvg[j] / (N_points - i - 1) - training_label[tmpIDs[tmpIDs.Count - i - 1]][j] / (N_points - i - 1);
                    rightErr[j] = rightErr[j] + (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - rightAvg[j]) * (training_label[tmpIDs[tmpIDs.Count - i - 1]][j] - rightAvg[j]) * Convert.ToDouble(i) / Convert.ToDouble(i + 1);
                    rightAvg[j] = rightAvg[j] * Convert.ToDouble(i) / Convert.ToDouble(i + 1) + training_label[tmpIDs[tmpIDs.Count - i - 1]][j] / Convert.ToDouble(i + 1);
                    tmp_err += leftErr[j] + rightErr[j];
                }

                if (lowest_err > tmp_err && training_dt[tmpIDs[tmpIDs.Count - i - 1]][dimIndex] != training_dt[tmpIDs[tmpIDs.Count - i - 2]][dimIndex]
                    && (i + 1) >= userConfig.minNodeSize && (i + userConfig.minNodeSize) < tmpIDs.Count)
                {
                    best_ID = tmpIDs[tmpIDs.Count - i - 1];
                    lowest_err = tmp_err;
                }
            }

            if (best_ID == -1)
            {
                error_n_point[0] = double.MaxValue;
                error_n_point[1] = double.MaxValue;
                return error_n_point;
            }

            error_n_point[0] = Math.Max(lowest_err, 0);
            error_n_point[1] = training_GridIndex_dt[best_ID][dimIndex];

            return error_n_point;
        }

        private bool GetGiniPartitionResult(ref int dimIndex, ref int Maingridindex, List<GeoWave> GeoWaveArr, int GeoWaveID, double Error, bool[] Dims2Take)
        {
            double[][] error_dim_partition = new double[2][];//information gain, maingridIndex
            error_dim_partition[0] = new double[training_dt[0].Count()];
            error_dim_partition[1] = new double[training_dt[0].Count()];

            //PARALLEL RUN - SEARCHING BEST PARTITION IN ALL DIMS
            if (userConfig.useParallel)
            {
                Parallel.For(0, training_dt[0].Count(), i =>
                {
                    if (Dims2Take[i])
                    {
                        double[] tmpResult = getGiniPartitionLargeDB(i, GeoWaveArr[GeoWaveID]);
                        error_dim_partition[0][i] = tmpResult[0];//information gain
                        error_dim_partition[1][i] = tmpResult[1];//maingridIndex                    
                    }
                    else
                    {
                        error_dim_partition[0][i] = double.MinValue;//information gain
                        error_dim_partition[1][i] = -1;//maingridIndex                    
                    }
                });
            }
            else
            {
                for (int i = 0; i < training_dt[0].Count(); i++)
                {
                    if (Dims2Take[i])
                    {
                        double[] tmpResult = getGiniPartitionLargeDB(i, GeoWaveArr[GeoWaveID]);
                        error_dim_partition[0][i] = tmpResult[0];//information gain
                        error_dim_partition[1][i] = tmpResult[1];//maingridIndex                    
                    }
                    else
                    {
                        error_dim_partition[0][i] = double.MinValue;//information gain
                        error_dim_partition[1][i] = -1;//maingridIndex                    
                    }
                }
            }

            dimIndex = Enumerable.Range(0, error_dim_partition[0].Count())
                .Aggregate((a, b) => (error_dim_partition[0][a] > error_dim_partition[0][b]) ? a : b); //maximal gain (>)

            if (error_dim_partition[0][dimIndex] <= 0)
                return false;//if best partition doesn't help - return

            Maingridindex = Convert.ToInt32(error_dim_partition[1][dimIndex]);
            return true;
        }

        private double[] getGiniPartitionLargeDB(int dimIndex, GeoWave geoWave)
        {
            double[] error_n_point = new double[2];//gain index
            if (wf.Program.MainGrid[dimIndex].Count == 1)//empty feature
            {
                error_n_point[0] = double.MinValue;//min gain
                error_n_point[1] = -1;
                return error_n_point;
            }
            //sort ids (for labels) acording to position at Form1.MainGrid[dimIndex][index]
            List<int> tmpIDs = new List<int>(geoWave.pointsIdArray);
            tmpIDs.Sort(delegate(int c1, int c2) { return training_dt[c1][dimIndex].CompareTo(training_dt[c2][dimIndex]); });

            if (training_dt[tmpIDs[0]][dimIndex] == training_dt[tmpIDs[tmpIDs.Count - 1]][dimIndex])//all values are the same 
            {
                error_n_point[0] = double.MinValue;//min gain
                error_n_point[1] = -1;
                return error_n_point;
            }

            Dictionary<double, double> leftcategories = new Dictionary<double, double>(); //double as counter to enable devision
            Dictionary<double, double> rightcategories = new Dictionary<double, double>(); //double as counter to enable devision
            for (int i = 0; i < tmpIDs.Count(); i++)
            {
                if (leftcategories.ContainsKey(training_label[tmpIDs[i]][0]))
                    leftcategories[training_label[tmpIDs[i]][0]] += 1;
                else
                    leftcategories.Add(training_label[tmpIDs[i]][0], 1);
            }
            double N_points = Convert.ToDouble(tmpIDs.Count);
            double initialGini = calcGini(leftcategories, N_points);
            double NpointsLeft = N_points;
            double NpointsRight = 0;
            double leftGini = 0;
            double rightGini = 0;
            double gain = 0;
            double bestGain = 0;
            int best_ID = -1;

            for (int i = 0; i < tmpIDs.Count - 1; i++)//we dont calc the last (rightmost) boundary - it equal to the left most
            {
                double rightMostLable = training_label[tmpIDs[tmpIDs.Count - i - 1]][0];

                if (leftcategories[rightMostLable] == 1)
                    leftcategories.Remove(rightMostLable);
                else
                    leftcategories[rightMostLable] -= 1;

                if (rightcategories.ContainsKey(rightMostLable))
                    rightcategories[rightMostLable] += 1;
                else
                    rightcategories.Add(rightMostLable, 1);

                NpointsLeft -= 1;
                NpointsRight += 1;

                leftGini = calcGini(leftcategories, NpointsLeft);
                rightGini = calcGini(rightcategories, NpointsRight);

                gain = (initialGini - leftGini) * (NpointsLeft / N_points) + (initialGini - rightGini) * (NpointsRight / N_points);

                if (gain > bestGain && training_dt[tmpIDs[tmpIDs.Count - i - 1]][dimIndex] != training_dt[tmpIDs[tmpIDs.Count - i - 2]][dimIndex]
                    && (i + 1) >= userConfig.minNodeSize && (i + userConfig.minNodeSize) < tmpIDs.Count 
                    )
                {
                    best_ID = tmpIDs[tmpIDs.Count - i - 1];
                    bestGain = gain;
                }
            }

            if (best_ID == -1)
            {
                error_n_point[0] = double.MinValue;//min gain
                error_n_point[1] = -1;
                return error_n_point;
            }

            error_n_point[0] = bestGain;
            error_n_point[1] = training_GridIndex_dt[best_ID][dimIndex];

            return error_n_point;
        }

        private double calcGini(Dictionary<double, double> Totalcategories, double Npoints)
        {
            double gini = 0;
            for (int i = 0; i < Totalcategories.Count; i++)
            {
                gini += (Totalcategories.ElementAt(i).Value / Npoints) * (1 - (Totalcategories.ElementAt(i).Value / Npoints));
            }
            return gini;
        }

        private bool getRandPartitionResult(ref int dimIndex, ref int Maingridindex, List<GeoWave> GeoWaveArr, int GeoWaveID, double Error, int seed=0)
        {
            Random rnd0 = new Random(seed);
            int seedIndex = rnd0.Next(0, Int16.MaxValue/2); 

            Random rnd = new Random(seedIndex + GeoWaveID);

            int counter = 0;
            bool partitionFound= false;

            while(!partitionFound && counter < 20)
            {
                counter++;
                dimIndex = rnd.Next(0, training_dt[0].Count()); // creates a number between 0 and GeoWaveArr[0].rc.dim 
                int partition_ID = GeoWaveArr[GeoWaveID].pointsIdArray[rnd.Next(1, GeoWaveArr[GeoWaveID].pointsIdArray.Count() - 1)];

                Maingridindex = Convert.ToInt32(training_GridIndex_dt[partition_ID][dimIndex]);//this is dangerouse for maingridIndex > 2^32
                
                return true;
            }

            return false;
        }

        private void setChildrensPointsAndMeanValue(ref GeoWave child0, ref GeoWave child1, int dimIndex, List<int> indexArr)
        {
            for (int i = 0; i < child0.MeanValue.Count(); i++)
            {
                child0.MeanValue[i] *=0;
                child1.MeanValue[i] *= 0;
            }

            //GO OVER ALL POINTS IN REGION
            for (int i = 0; i < indexArr.Count; i++)
            {
                if (training_dt[indexArr[i]][dimIndex] < wf.Program.MainGrid[dimIndex].ElementAt(child0.isotropicSplitsParameters.boundingBox[1][dimIndex]))
                {
                    for (int j = 0; j < training_label[0].Count(); j++)
                        child0.MeanValue[j] += training_label[indexArr[i]][j];
                    child0.pointsIdArray.Add(indexArr[i]);
                }
                else
                {
                    for (int j = 0; j < training_label[0].Count(); j++)
                        child1.MeanValue[j] += training_label[indexArr[i]][j];
                    child1.pointsIdArray.Add(indexArr[i]);
                }
            }

            for (int i = 0; i < child0.MeanValue.Count(); i++)
            {
                if (child0.pointsIdArray.Count > 0)
                    child0.MeanValue[i] /= Convert.ToDouble(child0.pointsIdArray.Count);
                if (child1.pointsIdArray.Count > 0)
                    child1.MeanValue[i] /= Convert.ToDouble(child1.pointsIdArray.Count);
            }
        }

        private void setChildrensPointsAndMeanValueSVMRegression(GeoWave child0, GeoWave child1, GeoWave parent, double[] svmApprox)
        {
            List<int> indexArr = parent.pointsIdArray;

            for (int i = 0; i < child0.MeanValue.Count(); i++)
            {
                child0.MeanValue[i] *= 0;
                child1.MeanValue[i] *= 0;
            }

            //GO OVER ALL POINTS IN REGION
            for (int i = 0; i < indexArr.Count; i++)
            {
                if (svmApprox[indexArr[i]] >= parent.svmRegressionSplitsParameters.svmRegressionSplitValue)
                {
                    for (int j = 0; j < training_label[0].Count(); j++)
                        child0.MeanValue[j] += training_label[indexArr[i]][j];
                    child0.pointsIdArray.Add(indexArr[i]);
                }
                else
                {
                    for (int j = 0; j < training_label[0].Count(); j++)
                        child1.MeanValue[j] += training_label[indexArr[i]][j];
                    child1.pointsIdArray.Add(indexArr[i]);
                }
            }

            for (int i = 0; i < child0.MeanValue.Count(); i++)
            {
                if (child0.pointsIdArray.Count > 0)
                    child0.MeanValue[i] /= Convert.ToDouble(child0.pointsIdArray.Count);
                if (child1.pointsIdArray.Count > 0)
                    child1.MeanValue[i] /= Convert.ToDouble(child1.pointsIdArray.Count);
            }
        }

        private double setChildrensPointsAndMeanValueLinearRegression(ref GeoWave child0, ref GeoWave child1, GeoWave parent)
        {
            List<int> indexArr = parent.pointsIdArray;

            //GO OVER ALL POINTS IN REGION
            for (int i = 0; i < indexArr.Count; i++)
            {
                double[] prediction = parent.linearRegressionSplitsParameters.linearRegression.Transform(training_dt[indexArr[i]]);
                if (prediction[0] >= 0)
                {
                    child0.pointsIdArray.Add(indexArr[i]);
                }
                else
                {
                    child1.pointsIdArray.Add(indexArr[i]);
                }
            }

            double[] child0Error = child0.calc_MeanValueReturnError(training_label, child0.pointsIdArray, ref child0.MeanValue);
            double[] child1Error = child1.calc_MeanValueReturnError(training_label, child1.pointsIdArray, ref child1.MeanValue);

            double totalError = 0;

            for (int i = 0; i < child0Error.Length; i++)
            {
                totalError += child0Error[i] + child1Error[i];
            }

            return totalError;
        }

        private double setChildrensPointsAndMeanValueSvmClassification(ref GeoWave child0, ref GeoWave child1, GeoWave parent)
        {
            List<int> indexArr = parent.pointsIdArray;

            //GO OVER ALL POINTS IN REGION
            for (int i = 0; i < indexArr.Count; i++)
            {
                bool prediction = parent.svmClassificationSplitParameters.svm.Decide(training_dt[indexArr[i]]);
                if (prediction)
                {
                    child0.pointsIdArray.Add(indexArr[i]);
                }
                else
                {
                    child1.pointsIdArray.Add(indexArr[i]);
                }
            }

            double[] child0Error = child0.calc_MeanValueReturnError(training_label, child0.pointsIdArray, ref child0.MeanValue);
            double[] child1Error = child1.calc_MeanValueReturnError(training_label, child1.pointsIdArray, ref child1.MeanValue);

            double totalError = 0;

            for (int i = 0; i < child0Error.Length; i++)
            {
                totalError += child0Error[i] + child1Error[i];
            }

            return totalError;
        }

        private bool[] getDim2Take( int Seed)
        {
            bool[] Dim2Take = new bool[training_dt[0].Count()];

            var ran = new Random(Seed);
            for (int i = 0; i < userConfig.nFeatures; i++)
            {
                //Dim2Take[dimArr[i]] = true;
                int index = ran.Next(0, training_dt[0].Count());
                if (Dim2Take[index] == true)
                    i--;
                else
                    Dim2Take[index] = true;
            }
            return Dim2Take;
        }
    }
}
