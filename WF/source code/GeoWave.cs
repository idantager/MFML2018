using System;
using System.Collections.Generic;
using System.Linq;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Statistics.Kernels;
using Accord.MachineLearning.VectorMachines;

namespace DataSetsSparsity
{
    public enum SplitType
    {
        NO_SPLIT,
        REGULAR_ISOTROPIC_SPLITS,
        GINI_INDEX_ISOTROPIC_CLASSIFICATION_SPLITS,
        LINEAR_REGRESSION_SPLITS,
        SVM_REGRESSION_SPLITS,
        SVM_CLASSIFICATION_SPLITS
    }

    public class IsotropicSplitsParameters
    {
        public int[][] boundingBox;
        public int dimIndex;
        public int maingridIndex;
        public double maingridValue;
        public int dimIndexSplitter;
        public double splitValue;

        public IsotropicSplitsParameters(int dataDim, int labelDim)
        {
            boundingBox = new int[2][];
            boundingBox[0] = new int[dataDim];
            boundingBox[1] = new int[dataDim];
            dimIndex = -1;
            maingridIndex = -1;
            maingridValue = -1;
        }
    }

    public class LinearRegressionSplitsParameters
    {
        public MultivariateLinearRegression linearRegression;
        public bool[] Dim2TakeNode;
    }

    public class SVMRegressionSplitsParameters
    {
        public SupportVectorMachine<Linear> svmRegression;
        public double svmRegressionSplitValue;
        public int labelIdx;
        public bool[] Dim2TakeNode;
    }

    public class SVMClassificationSplitParameters
    {
        public SupportVectorMachine<Gaussian> svm;
        public bool[] Dim2TakeNode;
    }

    public class GeoWave
    {
        public int ID;
        public int parentID, child0, child1, level;
        public double norm;
        public double[] MeanValue; //vector with means at each dimention
        public List<int> pointsIdArray = new List<int>();//points in regeion (row index of static input data)
        public SplitType splitType;
        public IsotropicSplitsParameters isotropicSplitsParameters;
        public LinearRegressionSplitsParameters linearRegressionSplitsParameters;
        public SVMRegressionSplitsParameters svmRegressionSplitsParameters;
        public SVMClassificationSplitParameters svmClassificationSplitParameters;
        public GeoWave(int dataDim, int labelDim)
        {
            Init(dataDim, labelDim);
        }

        private void Init(int dataDim, int labelDim)
        {
            parentID = -1;
            child0 = -1;
            child1 = -1;
            level = -1;
            norm = -1;
            MeanValue = new double[labelDim];
            ID = -1;
            splitType = SplitType.NO_SPLIT;
            isotropicSplitsParameters = new IsotropicSplitsParameters(dataDim, labelDim);
            svmRegressionSplitsParameters = new SVMRegressionSplitsParameters();
            linearRegressionSplitsParameters = new LinearRegressionSplitsParameters();
            svmClassificationSplitParameters = new SVMClassificationSplitParameters();
        }

        public GeoWave(int[][] BOX, int labelDim)
        {
            Init(BOX[0].Count(), labelDim);
            for (int j = 0; j < isotropicSplitsParameters.boundingBox[0].Count(); j++)
            {
                isotropicSplitsParameters.boundingBox[0][j] = BOX[0][j];
                isotropicSplitsParameters.boundingBox[1][j] = BOX[1][j];
            }        
        }

        public double[] calc_MeanValue(double[][] Labels_dt, List<int> indexArr)
        {
            double[] tmpMeanValue = new double[MeanValue.Count()];

            //GO OVER ALL POINTS IN REGION
            foreach (int index in indexArr)
            {
                for (int j = 0; j < Labels_dt[0].Count(); j++)
                    tmpMeanValue[j] += Labels_dt[index][j];
            }
            
            if (indexArr.Count * Labels_dt[0].Count() > 0)
            {
                for (int i = 0; i < tmpMeanValue.Count(); i++)
                {
                    tmpMeanValue[i] /= indexArr.Count;
                }
            }
            
            return tmpMeanValue;
        }
        //calculate sum of distances from mean vector
        public double calc_MeanValueReturnError(double[][] labelsDt, List<int> indexArr)
        {
            //NULLIFY
            double[] tmpMeanValue = calc_MeanValue(labelsDt, indexArr);
            double Error = 0;

            switch (userConfig.partitionType)
            {
                case "2": //L2
                    foreach (int index in indexArr)
                    {
                        for (int j = 0; j < labelsDt[0].Count(); j++)
                            Error += (labelsDt[index][j] - tmpMeanValue[j]) * (labelsDt[index][j] - tmpMeanValue[j]);//insert rc of norm type
                    }
                    return Error;
                case "1": //L1
                    foreach (int index in indexArr)
                    {
                        for (int j = 0; j < labelsDt[0].Count(); j++)
                            Error += Math.Abs(labelsDt[index][j] - tmpMeanValue[j]);
                    }
                    return Error;
                case "0":
                    foreach (int index in indexArr)
                    {
                        double sign = 0;
                        for (int j = 0; j < labelsDt[0].Count(); j++)
                            sign += Math.Abs(labelsDt[index][j] - tmpMeanValue[j]); ;
                        if (sign != 0)
                            Error += 1;
                    }
                    return Error;
                default:
                    {
                        double p = Convert.ToDouble(userConfig.partitionType);
                        //GO OVER ALL POINTS IN REGION
                        foreach (int index in indexArr)
                        {
                            for (int j = 0; j < labelsDt[0].Count(); j++)
                                Error += Math.Pow(labelsDt[index][j] - tmpMeanValue[j], p);
                        }
                        return Error;//same as l2 don't do sqrt  
                    }
            }
        }

        public double[] calc_MeanValueReturnError(double[][] labelsDt, List<int> indexArr, ref double[] calcedMeanValue)
        {
            //NULLIFY
            calcedMeanValue = calc_MeanValue(labelsDt, indexArr);
            double[] error = new double[labelsDt[0].Count()];

            switch (userConfig.partitionType)
            {
                case "2"://L2
                    foreach (int index in indexArr)
                    {
                        for (int j = 0; j < labelsDt[0].Count(); j++)
                            error[j] += (labelsDt[index][j] - calcedMeanValue[j]) * (labelsDt[index][j] - calcedMeanValue[j]);//insert rc of norm type
                    }
                    return error;
                case "1"://L1
                    foreach (int index in indexArr)
                    {
                        for (int j = 0; j < labelsDt[0].Count(); j++)
                            error[j] += Math.Abs(labelsDt[index][j] - calcedMeanValue[j]);
                    }
                    return error;
                case "0":          
                    for (int i = 0; i < indexArr.Count; i++)
                    {
                        double sign = 0;
                    }
                    return error;
                default:
                    {
                        double p = Convert.ToDouble(userConfig.partitionType);
                        //GO OVER ALL POINTS IN REGION
                        for (int i = 0; i < indexArr.Count; i++)
                        {
                            for (int j = 0; j < labelsDt[0].Count(); j++)
                                error[j] += Math.Pow(labelsDt[indexArr[i]][j] - calcedMeanValue[j], p);
                        }

                        return error;//same as l2 don't do sqrt  
                    }
            }
        }
        public void computeNormOfConsts(GeoWave parent, double Lp)
        {
            norm = 0;
            //GO OVER ALL POINTS IN THE REGION
            if (Lp == 2)
            {
                for (int j = 0; j < MeanValue.Count(); j++)
                    norm += ((MeanValue[j] - parent.MeanValue[j]) * (MeanValue[j] - parent.MeanValue[j]));
                norm *= pointsIdArray.Count();
                norm = Math.Sqrt(norm);
            }
            else if (Lp == 1)
            {
                for (int j = 0; j < MeanValue.Count(); j++)
                    norm += Math.Abs(MeanValue[j] - parent.MeanValue[j]);
                norm *= pointsIdArray.Count();
            }
            else
            {
                for (int j = 0; j < MeanValue.Count(); j++)
                    norm += Math.Pow(MeanValue[j] - parent.MeanValue[j], Lp);
                norm = norm * pointsIdArray.Count();
                norm *= pointsIdArray.Count();
            }
        }
        public void computeNormOfConsts(double Lp)
        {
            norm = 0;
            if (Lp == 2)
            {
                for (int j = 0; j < MeanValue.Count(); j++)
                    norm += (MeanValue[j] * MeanValue[j]);
                norm *= pointsIdArray.Count();
                norm = Math.Sqrt(norm);
            }
            else if (Lp == 1)
            {
                for (int j = 0; j < MeanValue.Count(); j++)
                    norm += Math.Abs(MeanValue[j]);
                norm *= pointsIdArray.Count();
            }
            else
            {
                for (int j = 0; j < MeanValue.Count(); j++)
                    norm += Math.Pow(MeanValue[j], Lp);
                norm *= pointsIdArray.Count();
                norm = Math.Pow(norm, 1 / Lp);
            }
            
        }
    }
}
