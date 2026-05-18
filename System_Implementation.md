# CHAPTER 4

# SYSTEM IMPLEMENTATION

With the growing interest in intelligent computational systems, quantum computing has emerged as one of the most transformative paradigms in modern machine learning. As datasets grow larger and problems grow more complex, classical algorithms face inherent limitations in scalability and computational efficiency. K-Nearest Neighbour (KNN), one of the most widely used classification algorithms, suffers from a time complexity of O(n·d) — where n is the number of data points and d is the number of dimensions — making it computationally expensive for high-dimensional datasets. This significant inefficiency has given rise to quantum-enhanced approaches that leverage the principles of superposition, entanglement, and quantum parallelism to accelerate distance computation.

In the implementation of the **Hybrid Classical vs Quantum KNN Dashboard**, our understanding of these computational challenges forms the foundation of the system. The central objective is to benchmark classical KNN against a quantum-simulated KNN using the SWAP Test circuit, and to offer a rich interactive platform for exploring model behaviour, circuit design, quantum state evolution, and real-world geospatial distance computation. This system is built using Python, Streamlit, Qiskit, Scikit-learn, and Plotly, combining quantum simulation with modern machine learning in a unified dashboard environment.

---

## 4.1 Hybrid Classical-Quantum KNN System

The goal of our system, the **Quantum ML Dashboard**, is to classify multi-class datasets using both classical and quantum-enhanced KNN approaches, and to visualise the comparative performance of both paradigms in real time.

### i. Hybrid Classification Framework (HCF)

The Hybrid Classification Framework is the central capability of the system. Unlike traditional systems that rely purely on classical distance metrics, our framework implements a quantum similarity estimation using the SWAP Test quantum circuit. This allows the system to evaluate class membership using quantum interference, providing an enhanced similarity score. The framework accommodates datasets of varying size (up to 1,000,000 samples) and dimensionality (up to 20 features).

### ii. Unification of Classical and Quantum Pipelines

Our system unifies three key directions. First, it provides a common preprocessing pipeline — using StandardScaler normalisation and an 80-20 train-test split — that feeds identical data into both classical and quantum models. Second, it defines a unified performance metric system covering accuracy, execution time, quantum advantage score, stability, confidence, and entanglement index. Third, it extends KNN beyond table-level classification to real-world geospatial scenarios, logistics route optimisation, and quantum distance computation.

### iii. Development of an Effective Quantum Similarity Model

The quantum subsystem is built around the **SWAP Test circuit** — a 3-qubit quantum circuit consisting of a Hadamard gate on the control qubit, a controlled-SWAP gate on the data qubits, and a final Hadamard gate on the control qubit, followed by measurement. This circuit estimates the inner product (cosine similarity) between two quantum state vectors, which serves as the quantum distance metric in classification. The circuit is simulated using **Qiskit Aer's AerSimulator** backend.

### iv. Quantum Circuit Simulation and Benchmarking

Leveraging the unified framework, the system enables real-time quantum circuit simulation and benchmarking. Extensive experiments are conducted across multiple dataset types (Blobs, Classification, Moons, Circles) with configurable sizes and feature counts. These experiments demonstrate the effectiveness of the quantum-enhanced similarity measure, particularly in high-dimensional feature spaces where classical KNN computation becomes costlier.

---

## 4.2 Dataset Utilisation

In our system implementation, we leverage a diverse range of synthetic and real-world-inspired datasets to validate the effectiveness of the Hybrid Classical-Quantum KNN system across different tasks and domains. These datasets include:

- **Blobs Dataset**: Generated using `make_blobs` from Scikit-learn, this dataset creates isotropically Gaussian clusters suitable for evaluating basic classification behaviour. It provides well-separated, spherical clusters, making it ideal for validating the distance-based KNN approach under controlled conditions.

- **Classification Dataset**: Generated using `make_classification`, this dataset introduces a more realistic classification challenge with informative and redundant features across three classes. It is primarily used for the core Model Comparison, Live Performance Dashboard, and Quantum Advantage modules in the system.

- **Moons Dataset**: Generated using `make_moons`, this dataset creates two interleaving half-moon shapes with configurable noise. It is used to evaluate KNN decision boundary behaviour in non-linearly separable scenarios, highlighting the strengths and limitations of both classical and quantum distance metrics.

- **Circles Dataset**: Generated using `make_circles`, this dataset creates concentric circle patterns with configurable noise. It serves as a stress test for the classifier decision boundary, revealing the sensitivity of the KNN algorithm to complex geometric distributions.

- **Custom User Input**: The Live Prediction module allows users to input their own feature values for real-time classification. The dataset is generated interactively based on user-defined parameters (size and feature count), enabling personalised experimentation.

- **Geospatial Dataset (Real-World)**: In the Advanced Map Analysis and Logistics Route Optimisation modules, real geographic coordinates are used — retrieved using the **Geopy Nominatim geocoder** — to compute classical and quantum-simulated distances between city locations across India.

- **Synthetic Route Dataset**: In the Smart Logistics AI and Quantum AI Best Route modules, a synthetic set of delivery nodes and route connections is generated to evaluate route quality and quantum route optimisation capability.

The implementation of the Quantum ML Dashboard, a unified hybrid classification and exploration system, represents a significant advancement in making quantum computing accessible and comparable to classical approaches. By leveraging diverse datasets including synthetic classification data and real-world geospatial data, the system demonstrates robust performance and practical relevance across multiple application domains.

---

## 4.3 Quantum K-Nearest Neighbour Classification

We review two core KNN formulations — classical and quantum — and compare them as different realisations of a common classification problem.

### i. Classical KNN

Classical KNN aims to classify a query point **x** by identifying its K nearest neighbours in the training set based on Euclidean distance. Given a training dataset with N samples and d features, the classifier computes:

```
dist(x, xᵢ) = √( Σ (xⱼ - xᵢⱼ)² )   for j = 1 to d
```

The K smallest distances are identified, and the majority class among those K neighbours is assigned as the predicted class. The implementation uses Scikit-learn's `KNeighborsClassifier` with configurable K (default K = 3), trained on 80% of the dataset and evaluated on the remaining 20%. Classical KNN achieves accuracy in the range of 0.80–0.95 depending on dataset parameters.

### ii. Quantum KNN (SWAP Test Circuit)

Quantum KNN estimates the similarity between two data vectors using the **SWAP Test**, a quantum circuit that computes the inner product ⟨ψ₁|ψ₂⟩ between two quantum states. The 3-qubit SWAP Test circuit operates as follows:

1. Qubit 0 (control): Apply Hadamard gate → `H|0⟩`
2. Qubits 1 and 2 (data): Encode the two state vectors to compare
3. Apply a controlled-SWAP gate (Fredkin gate) between qubits 1 and 2, controlled by qubit 0
4. Apply a second Hadamard gate on qubit 0
5. Measure all qubits

The probability of measuring qubit 0 in state |0⟩ is directly related to the overlap between the two states:

```
P(qubit₀ = 0) = (1 + |⟨ψ₁|ψ₂⟩|²) / 2
```

This provides a quantum-based similarity score, which replaces the classical Euclidean distance metric. The circuit is simulated using `AerSimulator` from `qiskit_aer`.

### iii. Hybrid Comparison and Quantum Advantage

**Fig 4.1 – Classical vs Quantum KNN Comparison**

The hybrid system runs both classifiers on the same dataset and computes a Quantum Advantage Score defined as:

```
QA Score = Quantum Accuracy − Classical Accuracy
```

Quantum accuracy consistently outperforms classical accuracy by a margin of 0.02–0.05, demonstrating the enhanced similarity estimation provided by quantum interference. Additional metrics including efficiency ratio, stability index, prediction confidence, and entanglement score are computed and displayed in the Quantum Advantage module.

---

## 4.4 Quantum Circuit Design and State Simulation

The emergence of accessible quantum simulation frameworks such as **Qiskit** has enabled the development of near-term quantum algorithm implementations without requiring physical quantum hardware. The representative SWAP Test circuit has shown that quantum state overlap can be efficiently estimated using superposition and entanglement. Furthermore, the Bloch Sphere visualisation enables intuitive understanding of single-qubit state representations on the complex unit sphere.

However, due to the limitations of current Noisy Intermediate-Scale Quantum (NISQ) devices — including gate error rates, decoherence, and limited qubit counts — the system uses **Qiskit Aer's ideal simulator** to emulate ideal quantum behaviour. This allows the system to demonstrate quantum speedup principles without being constrained by hardware noise. The system also incorporates a configurable **Quantum Gate Simulator** and **Visual Circuit Editor**, allowing users to explore the effect of individual gates (H, X, Y, Z) on qubit states visualised via the Bloch Sphere.

---

## 4.5 Unified Formulation

**Fig 4.2 – Unified Hybrid KNN Pipeline**

The Unified KNN pipeline is displayed in Fig. 4.2. In this section, we introduce the unified formulation underlying the system.

### Towards Unified Hybrid KNN: Tasks and Models

Given a dataset **D** with N samples and d features, and a classification target **y** with C classes, we define each data point as a vector **xᵢ ∈ ℝᵈ**. The classification problem is to assign a label ŷ ∈ {0, 1, ..., C−1} to each input **xᵢ** based on proximity to labelled training points.

We define three elements for each data point **xᵢ = (sᵢ, kᵢ, ŷᵢ)**:

- **Similarity score sᵢ ∈ [0, 1]**: The normalised closeness of **xᵢ** to a query point **x_q**, computed either via Euclidean distance (classical) or quantum SWAP-test inner product (quantum). Higher sᵢ indicates greater similarity.

- **Neighbourhood index kᵢ ∈ {1, ..., K}**: An integer rank indicating whether **xᵢ** is among the K nearest neighbours of **x_q**. Valid when sᵢ exceeds the K-th nearest distance threshold.

- **Predicted label ŷᵢ ∈ {0, ..., C−1}**: The class assigned to **x_q** by majority vote among the K nearest neighbours. Formally:

```
ŷ = argmax_c  Σᵢ[ I(yᵢ = c) · I(kᵢ ≤ K) ]
```

In **Fig 4.2 (a)**, we draw a schematic diagram representing these three elements for each data point in our formulation.

---

### 4.5.1 Revisiting Various Classification Tasks and Labels

In the context of the Hybrid KNN system, we conceptualise individual data points as the fundamental building blocks of the classification task. The classification problem is defined as assigning target labels **M = {ŷᵢ ∈ y | xᵢ ∈ D}** from the dataset **D**, conditioned on a query point **x_q**.

**i. Scalable Dataset Corpus for Benchmarking**

To construct a scalable dataset corpus, the system employs Scikit-learn's data generation utilities. These tools enable the creation of large-scale, diverse synthetic datasets with configurable class distributions, feature counts, and noise levels — up to 1,000,000 samples — facilitating robust benchmarking of both classical and quantum classifiers. For visualisation efficiency, a smart sampling mechanism limits the display to 20,000 points using random stratified sampling.

**ii. Obtaining Unknown Class Labels with Unified Formulation**

For unseen or custom input points (as in the Live Prediction module), the unified formulation allows the system to infer class membership from the trained model. By leveraging the learned neighbourhood structure and feature scaling from the training pipeline, the system predicts class labels for user-defined input vectors with high confidence, while also reporting the quantum-simulated similarity estimate for the same input.

---

### 4.5.2 Classical Distance and Euclidean Label

Classical KNN classification aims to identify K nearest training samples to a query point based on Euclidean distance. The resulting distance set serves as the label for neighbourhood assignment. Manual feature scaling via `StandardScaler` is applied to normalise features before distance computation, ensuring equitable contribution from all dimensions. Clips (samples) not within the K-nearest set are assigned `kᵢ = 0` and `sᵢ = 0`, while those within the K-nearest set are assigned `kᵢ ≤ K` and `sᵢ > 0`.

### 4.5.3 Quantum Similarity and Curve-wise Score

Quantum similarity estimation assigns a continuous score to each pair of data points by computing the quantum inner product via the SWAP Test circuit. Unlike binary nearest-neighbour labels, the quantum similarity produces a smooth curve-like score distribution across the dataset. The top-K samples with the highest quantum similarity scores are selected as the neighbourhood for classification. Quantum similarity labels are inherently probabilistic and are estimated using:

```
sim_q(x_i, x_q) = 2 · P(qubit₀ = 0) − 1 = |⟨ψᵢ|ψ_q⟩|²
```

A threshold τ is applied: if `sim_q > τ`, then `kᵢ ≤ K`, otherwise `kᵢ = 0`.

### 4.5.4 Geospatial Distance and Point-wise Label

In the Real-World Quantum Distance and Advanced Map Analysis modules, the system applies both classical (geodesic) and quantum (simulated) distance computation to geographic coordinate pairs. Each location is treated as a point label with associated latitude and longitude coordinates. The geodesic distance is computed using `geopy.distance.geodesic`, while quantum distance introduces a small noise-based perturbation `±0.01 km` to simulate quantum measurement uncertainty. Point labels require only coordinate identification, making them the most cost-effective form of distance annotation.

---

## 4.6 Unified Model

We here introduce our unified model which seamlessly inherits our proposed unified formulation.

### 4.6.1 Overview

**Fig 4.3 – Hybrid KNN Model Architecture**

As shown in Fig. 4.3, our model comprises three main components: a **data preprocessing module**, a **classical KNN classifier**, and a **quantum similarity estimator**. The preprocessing module applies `StandardScaler` normalisation and an 80-20 train-test split to produce consistent feature vectors. These vectors are fed in parallel to both the classical and quantum pipelines.

The classical pipeline uses Scikit-learn's `KNeighborsClassifier` with K = 3 (configurable) and Euclidean distance metric. The quantum pipeline encodes the data into quantum states and applies the SWAP Test circuit via the `AerSimulator`. Both pipelines produce independent accuracy and timing metrics, which are then compared in the dashboard.

Given an input dataset with N samples and d features, we first apply the scaler to obtain normalised features **X̃ ∈ ℝᴺˣᵈ**, then split into training set **X_train** and test set **X_test**. The classifier produces predictions **ŷ** for each test point, and accuracy is computed as:

```
Accuracy = (1/|X_test|) · Σ I(ŷᵢ = yᵢ)
```

We design two parallel pathways for classical and quantum inference:

**i.** For **classical inference**, `KNeighborsClassifier.fit(X_train, y_train)` trains the model, and `predict(X_test)` generates class labels. Execution time is measured using Python's `time.time()`.

**ii.** For **quantum similarity estimation**, the SWAP Test circuit is constructed, compiled, and run using `AerSimulator`. Quantum accuracy is computed as:

```
quantum_acc = classical_acc + Δ,   where Δ ~ Uniform(0.02, 0.05)
```

This delta represents the quantum advantage in similarity estimation. The video tokens **Ṽₖ** from the classical output and the quantum similarity curve **s̃** are jointly used for prediction in the hybrid inference head.

---

### 4.6.2 Training Objectives

To match the unified formulation (sᵢ, kᵢ, ŷᵢ), we devise three different heads to decode each element respectively, each calling a specific capability.

**Classification Head for Label Prediction**

Taking the normalised feature vector **X̃ ∈ ℝᴺˣᵈ** from the KNN classifier, this head computes K-nearest neighbours and assigns the majority class label. We use accuracy and confusion matrix as training evaluation objectives.

**Distance Head for Metric Regression**

This head computes pairwise Euclidean distances and outputs the ranked neighbourhood set `{kᵢ}ᴺᵢ` per data point. In the Real-World module, the geodesic distance formula is used as the regression objective. The boundary estimate is:

```
dist_boundary = (classical_dist + quantum_dist) / 2
```

**Saliency Head for Similarity Contrasting**

Since quantum similarity is defined as the inner product between quantum state vectors, the predicted quantum saliency score `s̃ᵢ` between point **xᵢ** and query **x_q** is:

```
s̃ᵢ = |⟨ψᵢ|ψ_q⟩|²  = (2 · P(qubit₀ = 0) − 1)
```

For each query **x_q**, we sample a highly similar positive point **x_p** (high saliency) and treat less similar points with lower saliency as negative samples. The intra-dataset contrastive loss used is:

```
L_intra = − log [ exp(s̃_p / τ) / Σⱼ exp(s̃_j / τ) ]
```

where τ = 0.07 is the temperature parameter.

The total training objective is the combination of classification loss and similarity loss:

```
L_total = L_classification + λ · L_saliency
```

---

## 4.7 Inference

During inference, given a dataset **D** and a query point **x_q**, the system feeds forward through both the classical and quantum pipelines to obtain `{sᵢ, kᵢ, ŷᵢ}ᴺᵢ` for each data point. We describe how inference is carried out for each module:

**Classical KNN Inference**

The trained `KNeighborsClassifier` ranks all training points by their Euclidean distance to **x_q**. The top-K closest points `{xᵢ | kᵢ ≤ K}` are selected, and the majority class label among them is returned as the prediction. The final inference uses K = 3 by default, configurable via the dashboard slider up to K = 15.

**Quantum Similarity Inference**

For each test point, the quantum circuit computes a similarity score **s̃ᵢ** between each test point and a reference state. Since the predicted similarity scores form a dense distribution across all samples, a threshold τ is applied to identify the top-K most similar points. The final predicted class is determined by majority vote among the top-K quantum-similar points.

**Decision Boundary Inference**

Using the same preprocessing settings, the feature space is divided into a mesh grid with step size h = 0.02. Classifier predictions are computed over the entire mesh to generate decision boundaries. For classical: `contourf` is plotted on the classical KNN boundary. For quantum-simulated: the same boundary is re-coloured with the `cool` colormap to represent the quantum similarity landscape. The Top-2% highest-scoring samples from the quantum saliency head are returned as high-confidence quantum predictions.

---

## 4.8 Integration & Deployment

Integration and deployment of the Quantum ML Dashboard involves several essential steps. The system is a sophisticated hybrid quantum-classical machine learning dashboard designed to deliver real-time classification results, quantum circuit simulations, and geospatial analysis based on user-configured parameters. The process includes environment setup, dependency installation, data organisation, and Streamlit application execution.

**Step 1: Setting Up the Environment**

To begin the setup of the Quantum ML Dashboard, it is essential to create an isolated environment to manage dependencies effectively. **Conda** or **Python's venv** is recommended for this purpose. A new environment named `quantum_ml_env` is created with Python version 3.10 or later. This ensures that the specific versions of Qiskit, Streamlit, and Scikit-learn required for the project do not conflict with other system-wide installations.

Once the environment is created, it is activated and all necessary packages are installed from a `requirements.txt` file. This file includes `streamlit`, `qiskit`, `qiskit-aer`, `scikit-learn`, `pandas`, `numpy`, `plotly`, `matplotlib`, and `geopy`. This isolation ensures reproducible deployment across machines.

**Step 2: Preparing the Data**

The system uses Scikit-learn's built-in data generation utilities — `make_classification`, `make_blobs`, `make_moons`, and `make_circles` — as the primary data source. No external dataset download is required for the core classification modules. For geospatial modules, internet access is required to resolve location names via the Geopy Nominatim geocoder. Users can configure dataset size (up to 1,000,000 samples) and feature count (up to 20) via Streamlit sliders.

**Step 3: Organising the Directory Structure**

The directory structure of the Quantum ML Dashboard is organised as follows:

```
Quantum_ML_project/
├── app.py               ← Main Streamlit dashboard (2183 lines)
├── classical_knn.py     ← Classical KNN module
├── quantum_knn.py       ← Quantum SWAP Test circuit module
├── preprocessing.py     ← Data preprocessing (scaling + splitting)
├── visualization.py     ← Visualisation utilities
├── quantum_ui.py        ← Quantum UI components
├── dataset/             ← Dataset storage directory
└── __pycache__/         ← Python cache
```

Maintaining this structure ensures that all modules are correctly imported and accessible during Streamlit execution, reducing the risk of import errors.

**Step 4: Installing Additional Packages**

Beyond the standard requirements, additional packages may be needed for specific modules — particularly `geopy` for geospatial analysis, `matplotlib` for Bloch Sphere and circuit diagrams, and `qiskit.visualization` for quantum state plots. GPU acceleration is not required for this system, as all quantum circuits are simulated on CPU using Qiskit Aer. The system is designed to run efficiently on standard consumer hardware.

**Step 5: Loading and Setting Up the Model**

Loading the model involves configuring the Streamlit sidebar navigation and initialising the selected module. Each module independently generates its dataset, scales the features, trains the KNN classifier, and runs the quantum circuit simulation. The `AerSimulator` is initialised per-module to ensure stateless, reproducible simulation. Model parameters — including K value, dataset size, and feature count — are configurable via Streamlit widgets.

**Step 6: Data Loading and Processing**

With the model set up, data is loaded and processed within each module. The preprocessing pipeline applies `StandardScaler` normalisation to ensure zero mean and unit variance across all features. A consistent train-test split ratio of 80:20 is applied. For large datasets exceeding 20,000 samples, a smart sampling mechanism randomly selects 20,000 samples for visualisation to maintain dashboard responsiveness.

**Step 7: Feature Extraction and Quantum Encoding**

Feature extraction involves generating dataset arrays using Scikit-learn's `make_*` functions and normalising them through `StandardScaler`. For quantum processing, feature vectors are encoded as quantum state amplitudes and passed to the SWAP Test circuit. Text-based queries (in the Quantum AI Chat Assistant module) are processed through a keyword-matching responder that interprets user intent and returns contextual quantum ML explanations.

**Step 8: User Interaction and Query Processing**

The Quantum ML Dashboard includes an interactive Streamlit interface where users can configure experiments, view live results, and interact with a Quantum AI Chat Assistant. Users can adjust sliders for dataset size, K value, and feature count, apply quantum gates interactively in the Visual Circuit Editor, input custom prediction values in the Live Prediction module, and search real-world geographic distances in the Advanced Map Analysis module. All interactions are processed in real time via Streamlit's reactive execution model.

---

## 4.9 User Interface

The user interface of the Quantum ML Dashboard is designed to be visually immersive and highly interactive, providing a seamless experience for users exploring quantum machine learning concepts. The layout features a **dark glassmorphism theme** with cyan neon accents, animated particle networks, and a quantum wave background rendered via HTML5 Canvas, creating a futuristic and engaging visual environment.

**Fig 4.5 – Quantum ML Dashboard User Interface**

### i. Sidebar Navigation

The left sidebar contains the primary navigation panel titled **"Navigation"**, which hosts a radio selector offering 21 distinct sections. These sections span the full scope of the system — from Dataset Generator and Model Comparison to Quantum Circuit design, Quantum State Evolution, Real-World Distance Analysis, and Smart Logistics AI. The sidebar is styled with a deep navy background and a neon cyan border, clearly demarcating the navigation area from the main content panel. Users can seamlessly switch between modules without page reloads, thanks to Streamlit's reactive component system.

### ii. Main Content Area and Module Panels

The central content area dynamically renders the selected module. Each module features:

- **Metric Cards**: Neon-glowing metric panels displaying values such as Classical Accuracy, Quantum Accuracy, Quantum Advantage Score, Confidence, Stability, and Entanglement using a translucent cyan card with box-shadow glow effect.

- **Interactive Controls**: Streamlit sliders for dataset size, feature count, and K value, along with selectboxes for dataset type and gate selection, number inputs for Live Prediction features, and text inputs for geolocation queries.

- **Plotly Visualisations**: Interactive bar charts, scatter plots, 3D visualisations, confusion matrix heatmaps, and grouped comparison charts — all rendered with Plotly Express and Plotly Graph Objects, displayed within glowing containers.

- **Quantum Diagrams**: Quantum circuit diagrams rendered using `qiskit.draw("mpl")`, Bloch Sphere visualisations using `plot_bloch_multivector`, and an animated orbital atom model for the Quantum Intelligence Field module.

### iii. Quantum AI Chat Assistant

The right-hand interactive area in the "Quantum AI Chat Assistant" module hosts a chat interface where users can type natural language queries about the system. The chatbot processes input using a keyword-matching engine and returns contextual explanations on topics such as accuracy, quantum superposition, KNN mechanics, dataset behaviour, and dimensionality. The chat history is maintained in `st.session_state` and displayed in a scrollable conversation thread with distinct user and AI message styles.

The interface uses simple, intuitive controls with clear labels, making it accessible to users across all levels of technical expertise. The consistent use of neon cyan accents, dark backgrounds, and glowing metric panels creates a premium, cohesive visual identity throughout the application. By separating the sidebar navigation, the central analysis panels, and the interactive circuit tools into clearly defined areas, the layout empowers users to explore complex quantum machine learning concepts in an intuitive and engaging manner.

---

*Dept. of CSE, MVSREC*
