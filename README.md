# DRIP Statistical Learning

# As of 2.63 Drip is being transitioned for support and evolution by F+F. Please email lakshmi@synergicdesign.com for any other details while in transition.

**v2.63**  *1 March 2017*

DRIP Statistical Learning is a collection of Java libraries for Machine Learning and Statistical Evaluation.

DRIP Statistical Learning is composed of the following main libraries:

 * Probabilistic Sequence Measure Concentration Bounds Library
 * Statistical Learning Theory Framework Library
 * Empirical Risk Minimization Library
 * VC and Capacity Measure Library
 * Covering Numbers Library
 * Alternate Statistical Learning Library
 * Problem Space and Algorithms Families Library
 * Parametric Classification Library
 * Non-parametric Classification Library
 * Clustering Library
 * Ensemble Learning Library
 * Multi-linear Sub-space Learning Library
 * Real-Valued Sequence Learning Library
 * Real-Valued Learning Library
 * Sequence Labeling Library
 * Bayesian Library
 * Linear Algebra Suport Library

For Installation, Documentation and Samples, and the associated supporting Numerical Libraries please check out [DRIP] (https://github.com/lakshmiDRIP/DRIP).


##Features

###Probabilistic Bounds and Concentration of Measure Sequences
####Probabilistic Bounds
 * Tail Probability Bounds Estimation
 * Basic Probability Inequalities
 * Cauchy-Schwartz Inequality
 * Association Inequalities
 * Moment, Gaussian, and Exponential Bounds
 * Bounding Sums of Independent Random Variables
 * Non Moment Based Bounding - Hoeffding Bound
 * Moment Based Bounds
 * Binomial Tails
 * Custom Bounds for Special i.i.d. Sequences

####Efron Stein Bounds
 * Martingale Differences Sum Inequality
 * Efron-Stein Inequality
 * Bounded Differences Inequality
 * Bounded Differences Inequality - Applications
 * Self-Bounding Functions
 * Configuration Functions
 
####Entropy Methods
 * Information Theory - Basics
 * Tensorization of the Entropy
 * Logarithmic Sobolev Inequalities
 * Logarithmic Sobolev Inequalities - Applications
 * Exponential Inequalities for Self-Bounding Functions
 * Combinatorial Entropy
 * Variations on the Theme of Self-Bounding Functions

####Concentration of Measure
 * Equivalent Bounded Differences Inequality
 * Convex Distance Inequality
 * Convex Distance Inequality - Proof
 * Application of the Convex Distance Inequality - Bin Packing

###Statistical Learning Theory - Foundation and Framework
####Standard SLT Framework
 * Computational Learning Theory
 * Probably Approximately Correct (PAC) Learning
 * PAC Definitions and Terminology
 * SLT Setup
 * Algorithms for Reducing Over-fitting
 * Bayesian Normalized Regularizer Setup

####Generalization and Consistency
 * Types of Consistency
 * Bias-Variance or Estimation-Approximation Trade-off
 * Bias-Variance Decomposition
 * Bias-Variance Optimization
 * Generalization and Consistency for kNN

###Empirical Risk Minimization - Principles and Techniques
####Empirical Risk Minimization
 * Overview
 * The Loss Functions and Empirical Risk Minimization Principles
 * Application of the Central Limit Theorem (CLT) and the Law of Large Numbers (LLN)
 * Inconsistency of Empirical Risk Minimizers
 * Uniform Convergence
 * ERM Complexity

####Symmetrization
 * The Symmetrization Lemma

####Generalization Bounds
 * The Union Bound
 * Shattering Coefficient
 * Empirical Risk Generalization Bound
 * Large Margin Bounds

####Rademacher Complexity
 * Rademacher-based Uniform Convergence
 * VC Entropy
 * Chaining Technique

####Local Rademacher Averages
 * Star-Hull and Sub-Root Functions
 * Local Rademacher Averages and Fixed Point
 * Local Rademacher Averages - Consequences

####Normalized ERM
 * Computing the Normalized Empirical Risk Bounds
 * De-normalized Bounds

####Noise Conditions
 * SLT Analysis Metrics
 * Types of Noise Conditions
 * Relative Loss Class

###VC Theory and Capacity Measure Analysis
####VC Theory and VC Dimension
 * Empirical Processes
 * Bounding the Empirical Loss Function
 * VC Dimension - Setup
 * Incorporating the Formal VC Definition
 * VC Dimension Examples
 * VC Dimension vs. Popper's Dimension

####Sauer Lemma and VC Classifier Framework
 * Working out Sauer Lemma Bounds
 * Sauer Lemma ERM Bounds
 * VC Index
 * VC Classifier Framework

###Capacity/Complexity Estimation Using Covering Numbers
####Covering and Entropy Numbers
 * Nomenclature- Normed Spaces
 * Covering, Entropy, and Dyadic Numbers
 * Background and Overview of Basic Results

####Covering Numbers for Real-Valued Function Classes
 * Functions of Bounded Variation
 * Functions of Bounded Variation - Upper Bound
 * Functions of Bounded Variation - Lower Bound
 * General Function Classes
 * General Function Class Bounds
 * General Function Class Bounds - Lemmas
 * General Function Class - Upper Bounds
 * General Function Class - Lower Bounds

####Operator Theory Methods for Entropy Numbers
 * Generalization Bounds via Uniform Convergence
 * Basic Uniform Convergence Bounds
 * Loss Function Induced Classes
 * Standard Form of Uniform Convergence

####Kernel Machines
 * SVM Capacity Control
 * Nonlinear Kernels
 * Generalization Performance of Regularization Networks
 * Covering Number Determination Steps
 * Challenges Presenting Master Generalization Error

####Entropy Number for Kernel Machines
 * Mercer Kernels
 * Equivalent Kernels
 * Mapping Phi into L2
 * Corrigenda to the Mercer Conditions
 * L2 Unit Ball -> Epsilon Mapping Scaling Operator
 * Unit Bounding Operator Entropy Numbers
 * The SVM Operator
 * Maurey's Theorem
 * Bounds for SV Classes
 * Asymptotic Rates of Decay for the Entropy Numbers

####Discrete Spectra of Convolution Operators
 * Kernels with Compact/Non-compact Support
 * The Kernel Operator Eigenvalues
 * Choosing Nu
 * Extensions to d-dimensions

####Covering Numbers for Given Decay Rates
 * Asymptotic/Non-asymptotic Decay of Covering Numbers
 * Polynomial Eigenvalue Decay
 * Summation and Integration of Non-decreasing Functions
 * Exponential Polynomial Decay

####Kernels for High-Dimensional Data
 * Kernel Fourier Transforms
 * Degenerate Kernel Bounds
 * Covering Numbers for Degenerate Systems
 * Bounds for Kernels in R^d
 * Impact of the Fourier Transform Decay on the Entropy Numbers

####Regularization Networks Entropy Numbers Determination - Practice
 * Custom Applications of the Kernel Machines Entropy Numbers
 * Extensions to the Operator-Theoretic Viewpoint for Covering Numbers

###Alternate Statistical Learning Approaches
####Minimum Description Length Approach
 * Coding Approaches
 * MDL Analyses

####Bayesian Methods
 * Bayesian and Frequentist Approaches
 * Bayesian Approaches

####Knowledge Based Bounds
 * Places to Incorporate Bounds
 * Prior Knowledge into the Function Space

####Approximation Error and Bayes' Consistency
 * Nested Function Spaces
 * Regularization
 * Achieving Zero Approximation Error
 * Rate of Convergence

####No Free Lunch Theorem
 * Algorithmic Consistency
 * NFT Formal Statements

###Problem Space and Algorthm Families
####Generative and Discriminative Models
 * Generative Models
 * Discriminant Models
 * Examples of Discriminant Approaches
 * Differences between Generative and Discriminant Models

####Supervised Learning
 * Supervised Learning Practice Steps
 * Challenges with Supervised Learning Practice
 * Formulation

####Unsupervised Learning

####Machine Learning
 * Calibration vs. Learning

####Pattern Recognition
 * Supervised vs. Unsupervised Pattern Recognition
 * Probabilistic Pattern Recognition
 * Formulation of Pattern Recognition
 * Pattern Recognition Practice SKU
 * Pattern Recognition Applications

###Parametric Classification Algorithms
####Statistical Classification
####Linear Discriminant Analysis
 * Setup and Formulation
 * Fischer's Linear Discriminant
 * Quadratic Discriminant Analysis

####Logistic Regression
 * Formulation
 * Goodness of Fit
 * Mathematical Setup
 * Bayesian Logistic Regression
 * Logistic Regression Extensions
 * Model Suitability Tests with Cross Validation

####Multinomial Logistic Regression
 * Setup and Formulation

###Non-Parametric Classification Algorithms
####Decision Trees and Decision Lists
####Variable Bandwidth Kernel Density Estimation
####k Nearest Neighbors Algorithm
####Perceptron
####Support Vector Machines (SVM)
####Gene Expression Programming (GEP)

###Clustering Algorithms
####Cluster Analysis
 * Cluster Models
 * Connectivity Based Clustering
 * Centroid Based Clustering
 * Distribution Based Clustering
 * Density Based Clustering
 * Clustering Enhancements
 * Internal Cluster Evaluation
 * External Cluster Evaluation
 * Clustering Axiom

####Mixture Model
 * Generic Mixture Model Details
 * Specific Mixture Models
 * Mixture Model Samples
 * Identifiability
 * Expectation Maximization
 * Alternatives to Expectation Maximization
 * Mixture Model Extensions

####Deep Learning
 * Unsupervised Representation Learner
 * Deep Learning using ANN
 * Deep Learning Architectures
 * Challenges with the DNN Approach
 * Deep Belief Networks (DBN)
 * Convolutional Neural Networks (CNN)
 * Deep Learning Evaluation Data Sets
 * Neurological Basis of Deep Learning

####Hierarchical Clustering

####k-Means Clustering
 * Mathematical Formulation
 * The Standard Algorithm
 * k-Means Initialization Schemes
 * k-Means Complexity
 * k-Means Variations
 * k-Means Applications
 * Alternate k-Means Formulations

####Correlation Clustering

####Kernel Principal Component Analysis (Kernel PCA)

###Ensemble Learning Algorithms

####Ensemble Learning
 * Overview
 * Theoretical Underpinnings
 * Ensemble Aggregator Types
 * Bayes' Optimal Classifier
 * Bagging and Boosting
 * Bayesian Model Averaging (BMA)
 * Baysian Model Combination (BMC)
 * Bucket of Models (BOM)
 * Stacking
 * Ensemble Averaging vs. Basis Spline Representation

####ANN Ensemble Averaging
 * Techniques and Results

####Boosting
 * Philosophy behind Boosting Algorithms
 * Popular Boosting Algorithms and Drawbacks

####Bootstrap Averaging
 * Sample Generation
 * Bagging with 1NN - Theoretical Treatment

###Multi-linear Sub-space Learning Algorithms
####Tensors and Multi-linear Sub-space Algorithms
 * Tensors
 * Multi-linear Sub-space Learning
 * Multi-linear PCA

###Real-Valued Sequence Learning Algorithms
####Kalman Filtering
 * Continuous Time Kalman Filtering
 * Nonlinear Kalman Filtering
 * Kalman Smoothing

####Particle Filtering

###Real-Valued Learning Algorithms
####Regression Analysis
 * Linear Regression
 * Assumptions Underlying Basis Linear Regression
 * Multi-variate Regression Analysis
 * Multi-variate Predictor/Response Regression
 * OLS on Basis Spline Representation
 * OLS on Basis Spline Representation with Roughness Penalty
 * Linear Regression Estimator Extensions
 * Bayesian Approach to Regression Analysis

####Component Analysis
 * Independent Component Analysis (ICA) Specification
 * Independent Component Analysis (ICA) Formulation
 * Principal Component Analysis
 * Principal COmponent Analysis - Constrained Formulation
 * 2D Principal Component Analysis - Constrained Formulation
 * 2D Principal Component Analysis - Lagrange Multiplier Based Constrained Formulation
 * nD Principal Component Analysis - Lagrange Multiplier Based Constrained Formulation
 * Information Theoretic Analysis of PCA
 * Empirical PCA Estimation From Data Set

###Sequence Label Learning Algorithms
####Hidden Markov Models
 * HMM State Transition/Emission Parameter Estimation
 * HMM Based Inference
 * Non-Bayesian HMM Model Setup
 * Bayesian Extension to the HMM Model Setup
 * HMM in Practical World

####Markov Chain Models
 * Markov Property
 * Markov Chains
 * Classification of the Markov Models
 * Monte Carlo Markov Chains (MCMC)
 * MCMC for Multi-dimensional Integrals

####Markov Random and Conditional Fields
 * MRF/CRF Axiomatic Properties/Definitions
 * Clique Factorization
 * Inference in MRF/CRF

####Maximum Entropy Markov Models

####Probabilistic Grammar and Parsing
 * Parsing
 * Parser
 * Context-Free Grammar (CFG)

###Bayesian Analysis
####Concepts, Formulation, Usage, and Application
 * Applicability
 * Analysis of Bayesian Systems
 * Bayesian Networks
 * Hypothesis Testing
 * Bayesian Updating
 * Maximum Entropy Techniques
 * Priors
 * Predictive Posteriors and Priors
 * Approximate Bayesian Computation
 * Measurement and Parametric Calibration
 * Bayesian Regression Analysis
 * Extensions to Bayesian Regression Analysis
 * Spline Proxying of Bayesian Systems

###Linear Algebra Support
####Optimizer
 * Constrained Optimization using Lagrangian
 * Least Squares Optimizer
 * Multi-variate Distribution

####Linear Systems Analysis and Transformation
 * Matrix Transforms
 * System of Linear Equations
 * Orthogonalization
 * Gaussian Elimination


##Contact

lakshmi@synergicdesign.com
